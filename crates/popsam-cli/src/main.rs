use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use candle_core::Device;
use popsam_core::{
    run_election, CandleEmbeddingModelFiles, CandleEmbeddingModelSpec, CandleEmbeddingProvider,
    ElectionConfig, ElectionResult, EmbeddedTextInput, EmbeddingProvider, InputRecord,
    OpenAiCompatibleEmbeddingProvider,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "popsam")]
#[command(about = "Representative text sampling from embeddings")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Elect(ElectArgs),
    EmbedElect(EmbedElectArgs),
}

#[derive(Debug, Parser)]
struct ElectArgs {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = InputFormat::Jsonl)]
    format: InputFormat,
    #[arg(long, default_value_t = 10)]
    report_last_k: usize,
    #[arg(long, default_value_t = 0.05)]
    elimination_fraction: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    output: OutputFormat,
    #[arg(long)]
    include_text: bool,
    #[arg(long)]
    pretty: bool,
}

#[derive(Debug, Parser)]
struct EmbedElectArgs {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = TextInputFormat::Plain)]
    format: TextInputFormat,
    #[arg(long, value_enum, default_value_t = EmbeddingBackend::Local)]
    backend: EmbeddingBackend,
    #[arg(long, default_value_t = 10)]
    report_last_k: usize,
    #[arg(long, default_value_t = 0.05)]
    elimination_fraction: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long, default_value = "main")]
    revision: String,
    #[arg(long)]
    config_file: Option<PathBuf>,
    #[arg(long)]
    tokenizer_file: Option<PathBuf>,
    #[arg(long)]
    weights_file: Option<PathBuf>,
    #[arg(long)]
    api_base_url: Option<String>,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    api_model: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    output: OutputFormat,
    #[arg(long)]
    include_text: bool,
    #[arg(long)]
    pretty: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputFormat {
    Jsonl,
    Csv,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TextInputFormat {
    Jsonl,
    Csv,
    Plain,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EmbeddingBackend {
    Local,
    Openai,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Json,
    Table,
    Csv,
    Markdown,
}

#[derive(Debug, Deserialize)]
struct CsvEmbeddingRow {
    id: String,
    #[serde(default)]
    text: Option<String>,
    embedding: String,
}

#[derive(Debug, Deserialize)]
struct CsvTextRow {
    id: String,
    text: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Elect(args) => run_elect(args),
        Command::EmbedElect(args) => run_embed_elect(args),
    }
}

fn run_elect(args: ElectArgs) -> Result<()> {
    let records = match args.format {
        InputFormat::Jsonl => read_jsonl(args.input.as_ref())?,
        InputFormat::Csv => read_csv(args.input.as_ref())?,
    };

    let result = run_election(
        records,
        ElectionConfig {
            report_last_k: args.report_last_k,
            elimination_fraction: args.elimination_fraction,
            random_seed: args.seed,
        },
    )?;

    print_result(&result, args.output, args.pretty, args.include_text)?;
    Ok(())
}

fn run_embed_elect(args: EmbedElectArgs) -> Result<()> {
    let records = match args.format {
        TextInputFormat::Jsonl => read_text_jsonl(args.input.as_ref())?,
        TextInputFormat::Csv => read_text_csv(args.input.as_ref())?,
        TextInputFormat::Plain => read_text_plain(args.input.as_ref())?,
    };

    let embedded = match args.backend {
        EmbeddingBackend::Local => {
            let provider = build_local_provider(&args)?;
            provider.embed(&records)?
        }
        EmbeddingBackend::Openai => {
            let provider = build_openai_provider(&args)?;
            provider.embed(&records)?
        }
    };

    let result = run_election(
        embedded
            .into_iter()
            .map(|item| EmbeddedTextInput {
                id: item.id,
                text: item.text,
                embedding: item.embedding,
            })
            .collect(),
        ElectionConfig {
            report_last_k: args.report_last_k,
            elimination_fraction: args.elimination_fraction,
            random_seed: args.seed,
        },
    )?;

    print_result(&result, args.output, args.pretty, args.include_text)?;
    Ok(())
}

fn print_result(
    result: &ElectionResult,
    output: OutputFormat,
    pretty: bool,
    include_text: bool,
) -> Result<()> {
    match output {
        OutputFormat::Json => {
            if pretty {
                println!("{}", serde_json::to_string_pretty(result)?);
            } else {
                println!("{}", serde_json::to_string(result)?);
            }
        }
        OutputFormat::Table => {
            print_table(result, include_text);
        }
        OutputFormat::Csv => {
            print_csv(result, include_text);
        }
        OutputFormat::Markdown => {
            print_markdown(result, include_text);
        }
    }
    Ok(())
}

fn print_table(result: &ElectionResult, include_text: bool) {
    let text_by_id = text_lookup(result);
    println!("Winner: {}", result.winner_id);
    if include_text {
        if let Some(text) = text_by_id.get(&result.winner_id).and_then(|text| text.as_deref()) {
            println!("Winner Text: {}", text);
        }
    }
    println!(
        "Representative IDs: {}",
        result.representative_ids.join(", ")
    );
    if include_text {
        println!();
        println!("Representative Texts");
        for (index, candidate_id) in result.representative_ids.iter().enumerate() {
            println!("{:>3}. {}", index + 1, candidate_id);
            if let Some(text) = text_by_id.get(candidate_id).and_then(|text| text.as_deref()) {
                println!("     {}", text);
            }
        }
    }
    println!();
    println!("Final Ranking");
    for (index, candidate_id) in result.all_ranked_ids.iter().enumerate() {
        println!("{:>3}. {}", index + 1, candidate_id);
        if include_text {
            if let Some(text) = text_by_id.get(candidate_id).and_then(|text| text.as_deref()) {
                println!("     {}", text);
            }
        }
    }
    println!();
    println!("Last Rounds");
    for round in &result.rounds {
        println!(
            "Round {:>3} | active {:>3} | eliminated {}",
            round.round_index,
            round.active_candidates,
            if round.eliminated_candidate_ids.is_empty() {
                "-".to_string()
            } else {
                round.eliminated_candidate_ids.join(", ")
            }
        );
        println!("  {:<16} {:>6} {:>6} {:>6}", "candidate", "1st", "2nd", "3rd");
        for vote in &round.votes {
            println!(
                "  {:<16} {:>6} {:>6} {:>6}",
                truncate_id(&vote.id, 16),
                vote.first_votes,
                vote.second_votes,
                vote.third_votes
            );
            if include_text {
                if let Some(text) = text_by_id.get(&vote.id).and_then(|text| text.as_deref()) {
                    println!("  {}", text);
                }
            }
        }
        println!();
    }
}

fn print_csv(result: &ElectionResult, include_text: bool) {
    let text_by_id = text_lookup(result);
    println!("section,round_index,active_candidates,eliminated_candidate_ids,candidate_id,candidate_text,first_votes,second_votes,third_votes,rank,winner_id,winner_text,representative_ids");
    println!(
        "summary,,,,,,,,,,{},{},\"{}\"",
        escape_csv(&result.winner_id),
        optional_csv_text(include_text, text_by_id.get(&result.winner_id).cloned().flatten()),
        result.representative_ids.join(" ")
    );
    for (index, candidate_id) in result.representative_ids.iter().enumerate() {
        println!(
            "representative,,,,{},{},,,,{},,,",
            escape_csv(candidate_id),
            optional_csv_text(include_text, text_by_id.get(candidate_id).cloned().flatten()),
            index + 1
        );
    }
    for (index, candidate_id) in result.all_ranked_ids.iter().enumerate() {
        println!(
            "ranking,,,,{},{},,,,{},,,",
            escape_csv(candidate_id),
            optional_csv_text(include_text, text_by_id.get(candidate_id).cloned().flatten()),
            index + 1
        );
    }
    for round in &result.rounds {
        for vote in &round.votes {
            println!(
                "round,{},{},\"{}\",{},{},{},{},{},,,",
                round.round_index,
                round.active_candidates,
                round.eliminated_candidate_ids.join(" "),
                escape_csv(&vote.id),
                optional_csv_text(include_text, text_by_id.get(&vote.id).cloned().flatten()),
                vote.first_votes,
                vote.second_votes,
                vote.third_votes
            );
        }
    }
}

fn print_markdown(result: &ElectionResult, include_text: bool) {
    let text_by_id = text_lookup(result);
    println!("# Result");
    println!();
    println!("- Winner: `{}`", result.winner_id);
    if include_text {
        if let Some(text) = text_by_id.get(&result.winner_id).and_then(|text| text.as_deref()) {
            println!("- Winner Text: {}", text);
        }
    }
    println!(
        "- Representative IDs: {}",
        result
            .representative_ids
            .iter()
            .map(|id| format!("`{id}`"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();
    if include_text {
        println!("## Representative Texts");
        println!();
        for (index, candidate_id) in result.representative_ids.iter().enumerate() {
            println!("{}. `{}`", index + 1, candidate_id);
            if let Some(text) = text_by_id.get(candidate_id).and_then(|text| text.as_deref()) {
                println!("{}", text);
            }
            println!();
        }
    }
    println!("## Final Ranking");
    println!();
    if include_text {
        println!("| Rank | Candidate | Text |");
        println!("| ---: | :-------- | :--- |");
    } else {
        println!("| Rank | Candidate |");
        println!("| ---: | :-------- |");
    }
    for (index, candidate_id) in result.all_ranked_ids.iter().enumerate() {
        if include_text {
            println!(
                "| {} | `{}` | {} |",
                index + 1,
                candidate_id,
                markdown_text(text_by_id.get(candidate_id).and_then(|text| text.as_deref()))
            );
        } else {
            println!("| {} | `{}` |", index + 1, candidate_id);
        }
    }
    println!();
    println!("## Last Rounds");
    println!();
    for round in &result.rounds {
        println!(
            "### Round {}",
            round.round_index
        );
        println!();
        println!(
            "- Active candidates: {}",
            round.active_candidates
        );
        println!(
            "- Eliminated: {}",
            if round.eliminated_candidate_ids.is_empty() {
                "-".to_string()
            } else {
                round
                    .eliminated_candidate_ids
                    .iter()
                    .map(|id| format!("`{id}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        println!();
        if include_text {
            println!("| Candidate | Text | 1st | 2nd | 3rd |");
            println!("| :-------- | :--- | --: | --: | --: |");
        } else {
            println!("| Candidate | 1st | 2nd | 3rd |");
            println!("| :-------- | --: | --: | --: |");
        }
        for vote in &round.votes {
            if include_text {
                println!(
                    "| `{}` | {} | {} | {} | {} |",
                    vote.id,
                    markdown_text(text_by_id.get(&vote.id).and_then(|text| text.as_deref())),
                    vote.first_votes,
                    vote.second_votes,
                    vote.third_votes
                );
            } else {
                println!(
                    "| `{}` | {} | {} | {} |",
                    vote.id, vote.first_votes, vote.second_votes, vote.third_votes
                );
            }
        }
        println!();
    }
}

fn truncate_id(id: &str, width: usize) -> String {
    let mut output = String::new();
    for ch in id.chars().take(width.saturating_sub(1)) {
        output.push(ch);
    }
    if id.chars().count() >= width {
        output.push('~');
        output
    } else {
        id.to_string()
    }
}

fn escape_csv(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn optional_csv_text(include_text: bool, text: Option<String>) -> String {
    if include_text {
        escape_csv(text.as_deref().unwrap_or(""))
    } else {
        String::new()
    }
}

fn markdown_text(text: Option<&str>) -> String {
    text.unwrap_or("").replace('|', "\\|")
}

fn text_lookup(result: &ElectionResult) -> HashMap<String, Option<String>> {
    result
        .embeddings
        .iter()
        .map(|item| (item.id.clone(), item.text.clone()))
        .collect()
}

fn read_jsonl(path: Option<&PathBuf>) -> Result<Vec<EmbeddedTextInput>> {
    let reader = open_reader(path)?;
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(&line).context("invalid JSONL row")?);
    }
    Ok(records)
}

fn read_text_jsonl(path: Option<&PathBuf>) -> Result<Vec<InputRecord>> {
    let reader = open_reader(path)?;
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(&line).context("invalid JSONL text row")?);
    }
    Ok(records)
}

fn read_csv(path: Option<&PathBuf>) -> Result<Vec<EmbeddedTextInput>> {
    let mut csv_reader = csv::Reader::from_reader(open_read(path)?);
    let mut records = Vec::new();
    for row in csv_reader.deserialize::<CsvEmbeddingRow>() {
        let row = row?;
        let embedding = row
            .embedding
            .split_whitespace()
            .map(|value| value.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .context("invalid embedding value in CSV row")?;
        records.push(EmbeddedTextInput {
            id: row.id,
            text: row.text,
            embedding,
        });
    }
    Ok(records)
}

fn read_text_csv(path: Option<&PathBuf>) -> Result<Vec<InputRecord>> {
    let mut csv_reader = csv::Reader::from_reader(open_read(path)?);
    let mut records = Vec::new();
    for row in csv_reader.deserialize::<CsvTextRow>() {
        let row = row?;
        records.push(InputRecord {
            id: row.id,
            text: Some(row.text),
        });
    }
    Ok(records)
}

fn read_text_plain(path: Option<&PathBuf>) -> Result<Vec<InputRecord>> {
    let reader = open_reader(path)?;
    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(InputRecord {
            id: (index + 1).to_string(),
            text: Some(line),
        });
    }
    Ok(records)
}

fn build_local_provider(args: &EmbedElectArgs) -> Result<CandleEmbeddingProvider> {
    if let (Some(config), Some(tokenizer), Some(weights)) = (
        args.config_file.clone(),
        args.tokenizer_file.clone(),
        args.weights_file.clone(),
    ) {
        return Ok(CandleEmbeddingProvider::from_local_files(
            &CandleEmbeddingModelFiles {
                config,
                tokenizer,
                weights,
            },
            Device::Cpu,
            args.max_length,
        )?);
    }

    let mut spec = CandleEmbeddingModelSpec::multilingual_default();
    if let Some(model_id) = &args.model_id {
        spec.model_id = model_id.clone();
    }
    spec.revision = args.revision.clone();

    Ok(CandleEmbeddingProvider::from_hf_hub(
        &spec,
        Device::Cpu,
        args.max_length,
    )?)
}

fn build_openai_provider(args: &EmbedElectArgs) -> Result<OpenAiCompatibleEmbeddingProvider> {
    let base_url = args
        .api_base_url
        .clone()
        .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .context("missing API key; pass --api-key or set OPENAI_API_KEY")?;
    let model = args
        .api_model
        .clone()
        .or_else(|| std::env::var("OPENAI_EMBEDDING_MODEL").ok())
        .unwrap_or_else(|| "text-embedding-3-small".to_string());

    Ok(OpenAiCompatibleEmbeddingProvider::new(base_url, api_key, model))
}

fn open_reader(path: Option<&PathBuf>) -> Result<BufReader<Box<dyn Read>>> {
    Ok(BufReader::new(open_read(path)?))
}

fn open_read(path: Option<&PathBuf>) -> Result<Box<dyn Read>> {
    match path {
        Some(path) => Ok(Box::new(
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
        )),
        None => Ok(Box::new(io::stdin())),
    }
}
