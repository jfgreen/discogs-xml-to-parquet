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
    ListBuilder, StringArray, StringBuilder, StringDictionaryBuilder, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use arrow::record_batch::RecordBatch;

const READ_BUF_SIZE: usize = 1048576; // 1MB
const BATCH_SIZE: usize = 10000;

//TODO: Sort out unwraps
//TODO: Result type alias

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
        match &self {
            Event::End(e) if e.name().into_inner() == name.as_bytes() => true,
            _ => false,
        }
    }

    fn is_empty_tag(&self) -> bool {
        match &self {
            Event::Empty(_) => true,
            _ => false,
        }
    }
}

#[derive(Default)]
struct Release {
    id: u32,
    status: String,
    title: String,
    //TODO: Try something that avoids re-allocation for each item,
    genres: Vec<String>,
}

impl Release {
    fn clear(&mut self) {
        self.id = Default::default();
        self.status.clear();
        self.title.clear();
        self.genres.clear();
    }
}

//TODO: Things to try when it comes to writing lists
// 1) Store references in Release (probably wont work, as underlying buffer gets re-used from one field to the next)
// 2) Do the simple thing and have a list of strings in Release (re-allocate each time)
// 3) Find a way to re-use a chunk of memory for the list (arrow types? something else?)
// 4) Write straight to release batch writer? (but what about missing fields - rows become unaligned?)
// 5) Use rust native Vec of String, but re-use Strings manually

struct ReleaseBatchWriter {
    writer: ArrowWriter<File>,
    pending: usize,
    ids: UInt32Builder,
    statuses: StringDictionaryBuilder<Int8Type>,
    titles: StringBuilder,
    genres: ListBuilder<StringBuilder>,
    schema: Arc<Schema>,
}

impl ReleaseBatchWriter {
    fn new(output_file_path: &str) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            //TODO: Is dictionary encoding actually useful/working?
            Field::new(
                "status",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new("title", DataType::Utf8, false),
            //TODO: Should we dictionary encode this?
            Field::new_list("genres", Field::new("item", DataType::Utf8, true), false),
        ]));

        let writer_properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let output_file = File::create(output_file_path).unwrap();

        let writer =
            ArrowWriter::try_new(output_file, schema.clone(), Some(writer_properties)).unwrap();

        let status_values =
            StringArray::from(vec![Some("Accepted"), Some("Draft"), Some("Deleted")]);

        ReleaseBatchWriter {
            writer,
            pending: 0,
            ids: UInt32Builder::with_capacity(BATCH_SIZE),
            statuses: StringDictionaryBuilder::<Int8Type>::new_with_dictionary(3, &status_values)
                .unwrap(),
            genres: ListBuilder::new(StringBuilder::new()),
            titles: StringBuilder::with_capacity(BATCH_SIZE, 512),
            schema,
        }
    }

    fn add(&mut self, release: &Release) {
        self.ids.append_value(release.id);
        self.statuses.append_value(&release.status);
        self.titles.append_value(&release.title);
        self.genres.append_value(release.genres.iter().map(Some));
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
                            Arc::new(self.genres.finish()),
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

    let mut release_writer = ReleaseBatchWriter::new(&output_file_path);
    let mut release = Release::default();

    reader.advance()?.expect_start_of("releases")?;
    reader.advance()?.expect_new_line()?;

    let mut n: u128 = 0;

    //TODO: Nice abstraction that lets us iterate releases one at a time?
    loop {
        n += 1;
        let event = reader.advance()?;

        if event.is_end_of("releases") {
            break;
        }

        let release_start = event.expect_start_of("release")?;

        parse_release_attributes(&release_start, &mut release)?;
        parse_release(&mut reader, &mut release)?;

        release_writer.add(&release);
        release.clear();

        if n % 10_000 == 0 {
            println!("{n}");
        }
    }
    release_writer.close();

    reader.advance()?.expect_new_line()?;
    reader.advance()?.expect_eof()?;
    println!("DONE");

    Ok(())
}

fn parse_release_attributes(
    release_start: &BytesStart,
    release: &mut Release,
) -> Result<(), ProcessingError> {
    //TODO: deal with unwraps
    for a in release_start.attributes() {
        match a {
            Ok(Attribute {
                key: QName(b"id"),
                value: id,
            }) => {
                let id = std::str::from_utf8(&id).unwrap().parse().unwrap();
                release.id = id;
            }
            Ok(Attribute {
                key: QName(b"status"),
                value: status,
            }) => {
                let status = std::str::from_utf8(&status).unwrap();
                release.status.push_str(status);
            }
            _ => {} //TODO: This should be an error
        }
    }
    Ok(())
}

fn parse_release(reader: &mut EventReader, release: &mut Release) -> Result<(), ProcessingError> {
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
            b"title" => parse_title(reader, release)?,
            b"images" => parse_images(reader)?,
            b"artists" => parse_artists(reader)?,
            b"extraartists" => parse_extra_artists(reader)?,
            b"labels" => parse_labels(reader)?,
            b"formats" => parse_formats(reader)?,
            b"genres" => parse_genres(reader, release)?,
            b"country" => parse_country(reader)?,
            b"data_quality" => parse_data_quality(reader)?,
            b"master_id" => parse_master_id(reader)?,
            b"tracklist" => parse_tracklist(reader)?,
            b"videos" => parse_videos(reader)?,
            b"released" => parse_released(reader)?,
            b"companies" => parse_companies(reader)?,
            b"styles" => parse_styles(reader)?,
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

fn parse_title(reader: &mut EventReader, release: &mut Release) -> Result<(), ProcessingError> {
    let title = reader.advance()?.expect_text()?;

    release.title.push_str(std::str::from_utf8(&title).unwrap());

    reader.advance()?.expect_end_of("title")?;

    Ok(())
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

fn parse_artists(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse artists
    loop {
        let event = reader.advance()?;

        if event.is_end_of("artists") {
            break Ok(());
        }
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

fn parse_labels(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse labels
    loop {
        let event = reader.advance()?;

        if event.is_end_of("labels") {
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

fn parse_genres(reader: &mut EventReader, release: &mut Release) -> Result<(), ProcessingError> {
    loop {
        let event = reader.advance()?;

        if event.is_end_of("genres") {
            break Ok(());
        }

        event.expect_start_of("genre")?;

        let genre = reader.advance()?.expect_text()?.into_inner();
        let genre = std::str::from_utf8(&genre).unwrap();
        let genre = genre.replace("&amp;", "&");
        release.genres.push(String::from(genre));

        reader.advance()?.expect_end_of("genre")?;
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

fn parse_styles(reader: &mut EventReader) -> Result<(), ProcessingError> {
    //TODO: Parse styles
    loop {
        let event = reader.advance()?;

        if event.is_end_of("styles") {
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
