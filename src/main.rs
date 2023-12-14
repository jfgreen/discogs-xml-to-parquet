use flate2::bufread::GzDecoder;
use quick_xml::events::{attributes::Attribute, BytesStart, BytesText, Event};
use quick_xml::name::QName;
use quick_xml::reader::Reader;

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::ops::Deref;
use std::process;
use std::sync::Arc;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use arrow::array::{
    ListBuilder, StringArray, StringBuilder, StringDictionaryBuilder, StructBuilder, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Fields, Int8Type, Schema};
use arrow::record_batch::RecordBatch;

const READ_BUF_SIZE: usize = 1048576; // 1MB
const BATCH_SIZE: usize = 10000;

//TODO: Sort out unwraps -> actually throw specific errors
//TODO: Result type alias
//TODO: Consider macros for common attribute wrangling

#[derive(Debug)]
enum ProcessingError {
    ExpectedStart,
    ExpectedStartOf(&'static str),
    ExpectedEndOf(&'static str),
    ExpectedEmpty(&'static str),
    ExpectedText,
    ExpectedNewline,
    ExpectedEof,
    IoError(std::io::Error),
    XMLParseError(quick_xml::Error),
}

impl From<std::io::Error> for ProcessingError {
    fn from(err: std::io::Error) -> Self {
        ProcessingError::IoError(err)
    }
}

impl From<quick_xml::Error> for ProcessingError {
    fn from(err: quick_xml::Error) -> Self {
        ProcessingError::XMLParseError(err)
    }
}

struct EventReader {
    reader: Reader<BufReader<GzDecoder<BufReader<File>>>>,
    buf: Vec<u8>,
}

impl EventReader {
    fn new(file_path: String) -> Result<Self, ProcessingError> {
        let input_file = File::open(file_path)?;
        let input_file = BufReader::with_capacity(READ_BUF_SIZE, input_file);
        let input_file = GzDecoder::new(input_file);
        let input_file = BufReader::new(input_file);
        let reader = Reader::from_reader(input_file);
        let buf = Vec::new();
        Ok(EventReader { reader, buf })
    }

    fn advance(&mut self) -> Result<Event, ProcessingError> {
        self.buf.clear();
        let event = self.reader.read_event_into(&mut self.buf)?;
        Ok(event)
    }
}

trait EventExt<'a> {
    fn expect_start(self) -> Result<BytesStart<'a>, ProcessingError>;

    fn expect_start_of(self, name: &'static str) -> Result<BytesStart<'a>, ProcessingError>;

    fn expect_end_of(self, name: &'static str) -> Result<(), ProcessingError>;

    fn expect_empty(self, name: &'static str) -> Result<BytesStart<'a>, ProcessingError>;

    fn expect_new_line(self) -> Result<(), ProcessingError>;

    fn expect_text(self) -> Result<BytesText<'a>, ProcessingError>;

    fn expect_eof(self) -> Result<(), ProcessingError>;

    fn is_end_of(&self, end_of: &'static str) -> bool;

    fn is_empty_tag(&self) -> bool;
}

impl<'a> EventExt<'a> for Event<'a> {
    fn expect_start_of(self, name: &'static str) -> Result<BytesStart<'a>, ProcessingError> {
        match self {
            Event::Start(e) if e.name().into_inner() == name.as_bytes() => Ok(e),
            _ => Err(ProcessingError::ExpectedStartOf(name)),
        }
    }

    fn expect_start(self) -> Result<BytesStart<'a>, ProcessingError> {
        match self {
            Event::Start(e) => Ok(e),
            _ => Err(ProcessingError::ExpectedStart),
        }
    }

    fn expect_end_of(self, name: &'static str) -> Result<(), ProcessingError> {
        match self {
            Event::End(e) if e.name().into_inner() == name.as_bytes() => Ok(()),
            _ => Err(ProcessingError::ExpectedEndOf(name)),
        }
    }

    fn expect_empty(self, name: &'static str) -> Result<BytesStart<'a>, ProcessingError> {
        match self {
            Event::Empty(e) if e.name().into_inner() == name.as_bytes() => Ok(e),
            _ => Err(ProcessingError::ExpectedEmpty(name)),
        }
    }

    fn expect_new_line(self) -> Result<(), ProcessingError> {
        match self {
            Event::Text(e) if e.deref() == b"\n" => Ok(()),
            _ => Err(ProcessingError::ExpectedNewline),
        }
    }

    fn expect_text(self) -> Result<BytesText<'a>, ProcessingError> {
        match self {
            Event::Text(e) => Ok(e),
            _ => Err(ProcessingError::ExpectedText),
        }
    }

    fn expect_eof(self) -> Result<(), ProcessingError> {
        match self {
            Event::Eof => Ok(()),
            _ => Err(ProcessingError::ExpectedEof),
        }
    }

    fn is_end_of(&self, name: &'static str) -> bool {
        matches!(&self, Event::End(e) if e.name().into_inner() == name.as_bytes())
    }

    fn is_empty_tag(&self) -> bool {
        matches!(&self, Event::Empty(_))
    }
}

struct ReleaseBatchWriter {
    writer: ArrowWriter<File>,
    pending: usize,
    ids: UInt32Builder,
    statuses: StringDictionaryBuilder<Int8Type>,
    titles: StringBuilder,
    artists: ListBuilder<StructBuilder>,
    genres: ListBuilder<StringBuilder>,
    styles: ListBuilder<StringBuilder>,
    labels: ListBuilder<StructBuilder>,
    schema: Arc<Schema>,
}

impl ReleaseBatchWriter {
    fn new(output_file_path: &str) -> Self {
        let label_fields = Fields::from(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("cat_no", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
        ]);

        let artist_fields = Fields::from(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("anv", DataType::Utf8, true),
            Field::new("join", DataType::Utf8, true),
        ]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            //TODO: Is dictionary encoding actually useful/working?
            Field::new(
                "status",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new("title", DataType::Utf8, false),
            Field::new_list(
                "artists",
                Field::new_struct("item", artist_fields.clone(), true),
                false,
            ),
            //TODO: Should we dictionary encode genres and styles?
            //TODO: Can we verify which encoding is written?
            Field::new_list("genres", Field::new("item", DataType::Utf8, true), false),
            Field::new_list("styles", Field::new("item", DataType::Utf8, true), false),
            Field::new_list(
                "labels",
                Field::new_struct("item", label_fields.clone(), true),
                false,
            ),
        ]));

        let writer_properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let output_file = File::create(output_file_path).unwrap();

        let writer =
            ArrowWriter::try_new(output_file, schema.clone(), Some(writer_properties)).unwrap();

        let status_values =
            StringArray::from(vec![Some("Accepted"), Some("Draft"), Some("Deleted")]);

        //TODO: Review default capacities.

        ReleaseBatchWriter {
            writer,
            pending: 0,
            ids: UInt32Builder::with_capacity(BATCH_SIZE),
            statuses: StringDictionaryBuilder::<Int8Type>::new_with_dictionary(3, &status_values)
                .unwrap(),
            titles: StringBuilder::with_capacity(BATCH_SIZE, 512),
            artists: ListBuilder::new(StructBuilder::new(
                artist_fields,
                vec![
                    //TODO: This seems a bit fragile?
                    Box::new(StringBuilder::new()), // id
                    Box::new(StringBuilder::new()), // name
                    Box::new(StringBuilder::new()), // anv
                    Box::new(StringBuilder::new()), // join
                ],
            )),
            genres: ListBuilder::new(StringBuilder::new()),
            styles: ListBuilder::new(StringBuilder::new()),
            labels: ListBuilder::new(StructBuilder::new(
                label_fields,
                vec![
                    //TODO: This seems a bit fragile?
                    Box::new(StringBuilder::new()), // id
                    Box::new(StringBuilder::new()), // cat no
                    Box::new(StringBuilder::new()), // name
                ],
            )),
            schema,
        }
    }

    //TODO: For each of these "push" methods, we could record that this field,
    // for the current record, has been populated. This will let us check for
    // missing fields that would cause the columns to become un-aligned, but
    // more importantly supply a null / default

    fn push_id(&mut self, id: u32) {
        self.ids.append_value(id);
    }

    fn push_status(&mut self, status: &str) {
        self.statuses.append_value(status);
    }
    fn push_title(&mut self, title: &str) {
        self.titles.append_value(title);
    }

    //TODO: Is there a way to define the order of struct in one place, safely.

    fn push_artist_id(&mut self, id: &str) {
        self.artists
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value(id)
    }

    fn push_artist_name(&mut self, name: &str) {
        self.artists
            .values()
            .field_builder::<StringBuilder>(1)
            .unwrap()
            .append_value(name)
    }

    fn push_artist_anv(&mut self, anv: &str) {
        self.artists
            .values()
            .field_builder::<StringBuilder>(2)
            .unwrap()
            .append_value(anv)
    }

    fn push_artist_anv_null(&mut self) {
        self.artists
            .values()
            .field_builder::<StringBuilder>(2)
            .unwrap()
            .append_null()
    }

    fn push_artist_join(&mut self, join: &str) {
        self.artists
            .values()
            .field_builder::<StringBuilder>(3)
            .unwrap()
            .append_value(join)
    }

    fn push_artist_join_null(&mut self) {
        self.artists
            .values()
            .field_builder::<StringBuilder>(3)
            .unwrap()
            .append_null()
    }

    fn push_end_of_artist(&mut self) {
        self.artists.values().append(true)
    }

    fn push_genre(&mut self, genre: &str) {
        self.genres.values().append_value(genre);
    }

    fn push_style(&mut self, style: &str) {
        self.styles.values().append_value(style);
    }

    fn push_label_id(&mut self, id: &str) {
        //TODO: Is there a nicer way than this?
        self.labels
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value(id);
    }

    fn push_label_cat_no(&mut self, cat_no: &str) {
        //TODO: Is there a nicer way than this?
        self.labels
            .values()
            .field_builder::<StringBuilder>(1)
            .unwrap()
            .append_value(cat_no);
    }
    fn push_label_name(&mut self, name: &str) {
        //TODO: Is there a nicer way than this?
        self.labels
            .values()
            .field_builder::<StringBuilder>(2)
            .unwrap()
            .append_value(name);
    }

    fn push_end_of_label(&mut self) {
        self.labels.values().append(true);
    }

    fn write_release(&mut self) {
        // Mark end of current release in list builders
        self.artists.append(true);
        self.genres.append(true);
        self.styles.append(true);
        self.labels.append(true);

        self.pending += 1;

        if self.pending == BATCH_SIZE {
            self.flush()
        }
    }

    fn flush(&mut self) {
        if self.pending > 0 {
            self.writer
                .write(
                    &RecordBatch::try_new(
                        self.schema.clone(),
                        vec![
                            Arc::new(self.ids.finish()),
                            Arc::new(self.statuses.finish()),
                            Arc::new(self.titles.finish()),
                            Arc::new(self.artists.finish()),
                            Arc::new(self.genres.finish()),
                            Arc::new(self.styles.finish()),
                            Arc::new(self.labels.finish()),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
            self.pending = 0;
        }
    }

    fn close(mut self) {
        self.flush();
        self.writer.close().unwrap();
    }
}

fn main() -> Result<(), ProcessingError> {
    let (input_file_path, output_file_path) = read_args(env::args());
    let mut reader = EventReader::new(input_file_path)?;

    let mut writer = ReleaseBatchWriter::new(&output_file_path);

    reader.advance()?.expect_start_of("releases")?;
    reader.advance()?.expect_new_line()?;

    let mut n: u128 = 0;

    loop {
        n += 1;
        let event = reader.advance()?;

        if event.is_end_of("releases") {
            break;
        }

        let release_start = event.expect_start_of("release")?;

        parse_release_attributes(&release_start, &mut writer)?;
        parse_release(&mut reader, &mut writer)?;

        writer.write_release();

        if n % 10_000 == 0 {
            println!("{n}");
        }
    }
    writer.close();

    reader.advance()?.expect_new_line()?;
    reader.advance()?.expect_eof()?;
    println!("DONE");

    Ok(())
}

fn parse_release_attributes(
    release_start: &BytesStart,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    //TODO: deal with unwraps
    for a in release_start.attributes() {
        match a {
            Ok(Attribute {
                key: QName(b"id"),
                value: id,
            }) => {
                let id = std::str::from_utf8(&id).unwrap().parse().unwrap();
                writer.push_id(id);
            }
            Ok(Attribute {
                key: QName(b"status"),
                value: status,
            }) => {
                let status = std::str::from_utf8(&status).unwrap();
                writer.push_status(status);
            }
            _ => {} //TODO: This should be an error
        }
    }
    Ok(())
}

fn parse_release(
    reader: &mut EventReader,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    loop {
        let event = reader.advance()?;

        if event.is_end_of("release") {
            break;
        }

        if event.is_empty_tag() {
            continue;
        }

        let event = event.expect_start()?;

        // We can't assume the order of elements within a release
        match event.name().into_inner() {
            b"title" => parse_title(reader, writer)?,
            b"genres" => parse_genres(reader, writer)?,
            b"styles" => parse_styles(reader, writer)?,
            b"images" => parse_images(reader)?,
            b"artists" => parse_artists(reader, writer)?,
            b"extraartists" => parse_extra_artists(reader)?,
            b"labels" => parse_labels(reader, writer)?,
            b"formats" => parse_formats(reader)?,
            b"country" => parse_country(reader)?,
            b"data_quality" => parse_data_quality(reader)?,
            b"master_id" => parse_master_id(reader)?,
            b"tracklist" => parse_tracklist(reader)?,
            b"videos" => parse_videos(reader)?,
            b"released" => parse_released(reader)?,
            b"companies" => parse_companies(reader)?,
            b"notes" => parse_notes(reader)?,
            b"identifiers" => parse_identifiers(reader)?,
            _ => {
                dbg!(&event);
                dbg!(&event.name().into_inner());
                //TODO: Error instead
                panic!();
            }
        }
    }

    reader.advance()?.expect_new_line()?;

    Ok(())
}

fn parse_title(
    reader: &mut EventReader,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    let title = reader.advance()?.expect_text()?;
    let title = std::str::from_utf8(&title).unwrap();
    writer.push_title(title);

    reader.advance()?.expect_end_of("title")?;

    Ok(())
}

fn parse_genres(
    reader: &mut EventReader,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    loop {
        let event = reader.advance()?;

        if event.is_end_of("genres") {
            break Ok(());
        }

        event.expect_start_of("genre")?;

        let genre = reader.advance()?.expect_text()?.into_inner();
        let genre = std::str::from_utf8(&genre).unwrap();
        //TODO: Can we do this without an alloc?
        let genre = genre.replace("&amp;", "&");
        writer.push_genre(&genre);

        reader.advance()?.expect_end_of("genre")?;
    }
}

fn parse_styles(
    reader: &mut EventReader,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    loop {
        let event = reader.advance()?;

        if event.is_end_of("styles") {
            break Ok(());
        }

        event.expect_start_of("style")?;

        let style = reader.advance()?.expect_text()?.into_inner();
        let style = std::str::from_utf8(&style).unwrap();
        //TODO: Can we do this without an alloc?
        let style = style.replace("&amp;", "&");
        writer.push_style(&style);

        reader.advance()?.expect_end_of("style")?;
    }
}

fn parse_labels(
    reader: &mut EventReader,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    loop {
        let event = reader.advance()?;

        if event.is_end_of("labels") {
            break Ok(());
        }

        let label = event.expect_empty("label")?;

        for a in label.attributes() {
            match a {
                Ok(Attribute {
                    key: QName(b"id"),
                    value: id,
                }) => {
                    let id = std::str::from_utf8(&id).unwrap();
                    writer.push_label_id(id);
                }
                Ok(Attribute {
                    key: QName(b"catno"),
                    value: cat_no,
                }) => {
                    let cat_no = std::str::from_utf8(&cat_no).unwrap();
                    writer.push_label_cat_no(cat_no);
                }
                Ok(Attribute {
                    key: QName(b"name"),
                    value: name,
                }) => {
                    let name = std::str::from_utf8(&name).unwrap();
                    writer.push_label_name(name);
                }
                _ => {} //TODO: This should be an error
            }
        }

        writer.push_end_of_label();
    }
}

fn parse_artists(
    reader: &mut EventReader,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    loop {
        let event = reader.advance()?;

        if event.is_end_of("artists") {
            break Ok(());
        }

        event.expect_start_of("artist")?;

        parse_artist(reader, writer)?;
    }
}

fn parse_artist(
    reader: &mut EventReader,
    writer: &mut ReleaseBatchWriter,
) -> Result<(), ProcessingError> {
    loop {
        let event = reader.advance()?;

        if event.is_end_of("artist") {
            writer.push_end_of_artist();
            break Ok(());
        }

        let event = event.expect_start()?;

        match event.name().into_inner() {
            b"id" => {
                // Id should never be null
                let event = reader.advance()?;
                let id = event.expect_text()?;
                let id = std::str::from_utf8(&id).unwrap();
                writer.push_artist_id(id);
                reader.advance()?.expect_end_of("id")?;
            }
            b"name" => {
                // Name should never be null
                let event = reader.advance()?;
                let name = event.expect_text()?;
                let name = std::str::from_utf8(&name).unwrap();
                writer.push_artist_name(name);
                reader.advance()?.expect_end_of("name")?;
            }
            b"anv" => {
                // Artist name variation can be null
                let event = reader.advance()?;
                if event.is_end_of("anv") {
                    writer.push_artist_anv_null();
                } else {
                    let anv = event.expect_text()?;
                    let anv = std::str::from_utf8(&anv).unwrap();
                    writer.push_artist_anv(anv);
                    reader.advance()?.expect_end_of("anv")?;
                }
            }
            b"join" => {
                // Join field can be null
                let event = reader.advance()?;
                if event.is_end_of("join") {
                    writer.push_artist_join_null();
                } else {
                    let join = event.expect_text()?;
                    let join = std::str::from_utf8(&join).unwrap();
                    writer.push_artist_join(join);
                    reader.advance()?.expect_end_of("join")?;
                }
            }
            b"role" => {
                // Tracks never seems to hold a value for main artist, so we can skip
                reader.advance()?.expect_end_of("role")?;
            }
            b"tracks" => {
                // Tracks never seems to hold a value for main artist, so we can skip
                reader.advance()?.expect_end_of("tracks")?;
            }
            _ => {
                dbg!(&event);
                panic!("oh no"); //TODO: This should by an error
            }
        }
    }
}

fn parse_images(reader: &mut EventReader) -> Result<(), ProcessingError> {
    // Images are ignored because uris are not in the dataset
    loop {
        let event = reader.advance()?;

        if event.is_end_of("images") {
            break Ok(());
        }

        event.expect_empty("image")?;
    }
}

fn parse_extra_artists(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse extra artists
    loop {
        let event = reader.advance()?;

        if event.is_end_of("extraartists") {
            break Ok(());
        }
    }
}

fn parse_formats(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse formats
    loop {
        let event = reader.advance()?;

        if event.is_end_of("formats") {
            break Ok(());
        }
    }
}

fn parse_country(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse country
    loop {
        let event = reader.advance()?;

        if event.is_end_of("country") {
            break Ok(());
        }
    }
}

fn parse_data_quality(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse data quality
    loop {
        let event = reader.advance()?;

        if event.is_end_of("data_quality") {
            break Ok(());
        }
    }
}

fn parse_master_id(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse master_id
    loop {
        let event = reader.advance()?;

        if event.is_end_of("master_id") {
            break Ok(());
        }
    }
}

fn parse_tracklist(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse tracklist
    loop {
        let event = reader.advance()?;

        if event.is_end_of("tracklist") {
            break Ok(());
        }
    }
}

fn parse_videos(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse videos
    loop {
        let event = reader.advance()?;

        if event.is_end_of("videos") {
            break Ok(());
        }
    }
}

fn parse_released(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse released
    loop {
        let event = reader.advance()?;

        if event.is_end_of("released") {
            break Ok(());
        }
    }
}

fn parse_companies(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse companies
    loop {
        let event = reader.advance()?;

        if event.is_end_of("companies") {
            break Ok(());
        }
    }
}

fn parse_notes(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse notes
    loop {
        let event = reader.advance()?;

        if event.is_end_of("notes") {
            break Ok(());
        }
    }
}

fn parse_identifiers(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse identifiers
    loop {
        let event = reader.advance()?;

        if event.is_end_of("identifiers") {
            break Ok(());
        }
    }
}

fn read_args(args: impl IntoIterator<Item = String>) -> (String, String) {
    let mut args = args.into_iter();
    let exec = args.next();
    match (args.next(), args.next(), args.next()) {
        (Some(input_path), Some(output_path), None) => (input_path, output_path),
        _ => {
            let exec = exec.as_deref().unwrap_or("discogs-parquet");
            println!("Usage: {exec} input-file output-file");
            process::exit(1)
        }
    }
}
