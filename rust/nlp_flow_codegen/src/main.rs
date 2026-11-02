//! Render the Rust natural-language declarations for the Python package.

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use nlp_flow_core::{
    BenchmarkTarget, BenchmarkTargetKind, HierarchyLevels, OutputContracts, TypedChannel,
};
use nlp_flow_macros::{BenchmarkTargets, HierarchyLevels, OutputProfiles, TypedChannel};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum TextChannel {
    #[nlp_flow(name = "syntax")]
    Syntax,
    #[nlp_flow(name = "symbol")]
    Symbol,
    #[nlp_flow(name = "discourse")]
    Discourse,
    #[nlp_flow(name = "execution")]
    Execution,
    #[nlp_flow(name = "engineering")]
    Engineering,
}

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum CorpusFamily {
    #[nlp_flow(name = "general_language")]
    GeneralLanguage,
    #[nlp_flow(name = "code")]
    Code,
    #[nlp_flow(name = "model_architecture")]
    ModelArchitecture,
    #[nlp_flow(name = "engineering_acceleration")]
    EngineeringAcceleration,
}

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum CodeLanguage {
    #[nlp_flow(name = "python")]
    Python,
    #[nlp_flow(name = "rust")]
    Rust,
    #[nlp_flow(name = "cpp")]
    Cpp,
}

#[derive(Debug, Clone, Copy, TypedChannel, BenchmarkTargets)]
pub enum BenchmarkProxyTarget {
    #[nlp_flow(name = "compile_python", kind = "scalar")]
    CompilePython,
    #[nlp_flow(name = "compile_rust", kind = "scalar")]
    CompileRust,
    #[nlp_flow(name = "compile_cpp", kind = "scalar")]
    CompileCpp,
    #[nlp_flow(name = "execution_python", kind = "scalar")]
    ExecutionPython,
    #[nlp_flow(name = "execution_rust", kind = "scalar")]
    ExecutionRust,
    #[nlp_flow(name = "execution_cpp", kind = "scalar")]
    ExecutionCpp,
    #[nlp_flow(name = "runtime_success", kind = "scalar")]
    RuntimeSuccess,
    #[nlp_flow(name = "patch_apply", kind = "scalar")]
    PatchApply,
    #[nlp_flow(name = "repository_resolved", kind = "scalar")]
    RepositoryResolved,
    #[nlp_flow(name = "fail_to_pass", kind = "scalar")]
    FailToPass,
    #[nlp_flow(name = "pass_to_pass", kind = "scalar")]
    PassToPass,
    #[nlp_flow(name = "calibration", kind = "scalar")]
    Calibration,
    #[nlp_flow(name = "self_repair_success", kind = "scalar")]
    SelfRepairSuccess,
    #[nlp_flow(name = "line_completion", kind = "scalar")]
    LineCompletion,
    #[nlp_flow(name = "class_level_success", kind = "scalar")]
    ClassLevelSuccess,
    #[nlp_flow(name = "library_call_success", kind = "scalar")]
    LibraryCallSuccess,
    #[nlp_flow(name = "branch_coverage", kind = "scalar")]
    BranchCoverage,
    #[nlp_flow(name = "input_prediction_correct", kind = "scalar")]
    InputPredictionCorrect,
    #[nlp_flow(name = "output_prediction_correct", kind = "scalar")]
    OutputPredictionCorrect,
    #[nlp_flow(name = "efficiency_guard", kind = "scalar")]
    EfficiencyGuard,
    #[nlp_flow(name = "knowledge_correct", kind = "scalar")]
    KnowledgeCorrect,
    #[nlp_flow(name = "multiple_choice_correct", kind = "scalar")]
    MultipleChoiceCorrect,
    #[nlp_flow(name = "selective_correct", kind = "scalar")]
    SelectiveCorrect,
    #[nlp_flow(name = "answer_correctness", kind = "scalar")]
    AnswerCorrectness,
    #[nlp_flow(name = "informativeness", kind = "scalar")]
    Informativeness,
    #[nlp_flow(name = "truthfulness", kind = "scalar")]
    Truthfulness,
    #[nlp_flow(name = "safety_compliance", kind = "scalar")]
    SafetyCompliance,
    #[nlp_flow(name = "file_localization", kind = "token")]
    FileLocalization,
    #[nlp_flow(name = "patch_token", kind = "token")]
    PatchToken,
    #[nlp_flow(name = "trace_step", kind = "token")]
    TraceStep,
    #[nlp_flow(name = "trace_result", kind = "token")]
    TraceResult,
    #[nlp_flow(name = "line_completion_token", kind = "token")]
    LineCompletionToken,
    #[nlp_flow(name = "library_call_token", kind = "token")]
    LibraryCallToken,
    #[nlp_flow(name = "branch_execution_token", kind = "token")]
    BranchExecutionToken,
    #[nlp_flow(name = "test_output_token", kind = "token")]
    TestOutputToken,
    #[nlp_flow(name = "runtime_ratio", kind = "regression")]
    RuntimeRatio,
    #[nlp_flow(name = "memory_ratio", kind = "regression")]
    MemoryRatio,
    #[nlp_flow(name = "latency_ratio", kind = "regression")]
    LatencyRatio,
}

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum TrainingGroup {
    Base,
    Turn,
    Task,
    Dynamic,
    Channels,
    CrossAttention,
    Cfm,
    Json,
    GapAdapters,
    GapRouter,
    Proxy,
    Conversation,
    Factory,
    Growth,
    Unified,
    RuntimeCompetence,
    UnifiedPlannerShared,
    UnifiedDepthRoute,
    UnifiedLaneRoute,
    UnifiedCompetenceRoute,
    UnifiedBirthRoute,
    UnifiedBudgetRoute,
}

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum ArchitectureAxis {
    HiddenSize,
    NumLayers,
    IntermediateSize,
    DynamicMaxSteps,
    DynamicParallelLanes,
    DynamicParallelTopK,
    DynamicLaneRank,
    UnifiedPlannerDim,
    UnifiedCellRank,
    UnifiedTopK,
    UnifiedGeneratedRank,
    UnifiedGeneratedBasisCount,
    UnifiedGeneratedExpertMode,
    EnableDynamicStack,
    EnableUnifiedCompetence,
    EnableRuntimeCompetence,
    EnableJsonSchemaKeyHead,
    EnableTaskMetadata,
}

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum JsonAction {
    StartObject,
    EndObject,
    StartArray,
    EndArray,
    Key,
    String,
    Number,
    True,
    False,
    Null,
    Comma,
    Colon,
}

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum TreeDomain {
    Ast,
    Json,
    Patch,
    ToolAction,
}

#[derive(Debug, Clone, Copy, TypedChannel)]
pub enum OutputMode {
    Text,
    Code,
    Json,
    Ast,
}

#[derive(Debug, Clone, Copy, OutputProfiles)]
pub enum StructuredOutputProfile {
    #[nlp_flow(name = "text", structural = true, schema = false, executable = false)]
    Text,
    #[nlp_flow(name = "code", structural = true, schema = true, executable = true)]
    Code,
    #[nlp_flow(name = "json", structural = true, schema = true, executable = false)]
    Json,
    #[nlp_flow(name = "ast", structural = true, schema = true, executable = false)]
    Ast,
    #[nlp_flow(name = "patch", structural = true, schema = true, executable = true)]
    Patch,
    #[nlp_flow(
        name = "tool_action",
        structural = true,
        schema = true,
        executable = false
    )]
    ToolAction,
}

#[derive(Debug, Clone, Copy, HierarchyLevels)]
pub enum NlpHierarchy {
    #[nlp_flow(
        id = "L0",
        role = "global topic, intent, length, and required entities",
        max_state_tokens = 16
    )]
    DocumentPlan,
    #[nlp_flow(
        id = "L1",
        role = "paragraph, section, clause, and discourse roles",
        max_state_tokens = 128
    )]
    SegmentStructure,
    #[nlp_flow(
        id = "L2",
        role = "surface token and span flow state",
        max_state_tokens = 1024
    )]
    TokenSpanSurface,
    #[nlp_flow(
        id = "L3",
        role = "entities, relations, exact constraints, and verifier anchors",
        max_state_tokens = 256
    )]
    EntityConstraintAnchors,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Python,
}

fn python_string(value: &str) -> String {
    Value::String(value.to_owned()).to_string()
}

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn render_json() -> String {
    let proxy_names = |kind| {
        BenchmarkProxyTarget::ALL
            .iter()
            .filter(|target| target.kind() == kind)
            .map(|target| target.as_str())
            .collect::<Vec<_>>()
    };
    let payload = json!({
        "version": "nlp-flow-rust-contract-v1",
        "generated_by": "rust/nlp_flow_codegen",
        "typed_channels": TextChannel::ALL.iter().map(|channel| channel.as_str()).collect::<Vec<_>>(),
        "corpus_families": CorpusFamily::ALL.iter().map(|family| family.as_str()).collect::<Vec<_>>(),
        "code_languages": CodeLanguage::ALL.iter().map(|language| language.as_str()).collect::<Vec<_>>(),
        "benchmark_targets": {
            "scalar": proxy_names(BenchmarkTargetKind::Scalar),
            "token": proxy_names(BenchmarkTargetKind::Token),
            "regression": proxy_names(BenchmarkTargetKind::Regression),
        },
        "training_groups": TrainingGroup::ALL.iter().map(|group| group.as_str()).collect::<Vec<_>>(),
        "architecture_axes": ArchitectureAxis::ALL.iter().map(|axis| axis.as_str()).collect::<Vec<_>>(),
        "json_actions": JsonAction::ALL.iter().map(|action| action.as_str()).collect::<Vec<_>>(),
        "tree_domains": TreeDomain::ALL.iter().map(|domain| domain.as_str()).collect::<Vec<_>>(),
        "output_modes": OutputMode::ALL.iter().map(|mode| mode.as_str()).collect::<Vec<_>>(),
        "output_contracts": StructuredOutputProfile::CONTRACTS.iter().map(|contract| json!({
            "name": contract.name,
            "structural": contract.structural,
            "schema": contract.schema,
            "executable": contract.executable,
        })).collect::<Vec<_>>(),
        "hierarchy": NlpHierarchy::SPECS.iter().map(|spec| json!({
            "id": spec.id,
            "name": spec.name,
            "role": spec.role,
            "max_state_tokens": spec.max_state_tokens,
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string_pretty(&payload).expect("JSON rendering cannot fail") + "\n"
}

fn render_python() -> String {
    let mut output =
        String::from("\"\"\"Generated by rust/nlp_flow_codegen; do not edit by hand.\"\"\"\n\n");
    output.push_str("HIERARCHY_ROWS: tuple[tuple[str, str, str, int], ...] = (\n");
    for spec in NlpHierarchy::SPECS {
        output.push_str(&format!(
            "    ({}, {}, {}, {}),\n",
            python_string(spec.id),
            python_string(spec.name),
            python_string(spec.role),
            spec.max_state_tokens,
        ));
    }
    output.push_str(")\n\nTYPED_CHANNEL_NAMES: tuple[str, ...] = (\n");
    for channel in TextChannel::ALL {
        output.push_str(&format!("    {},\n", python_string(channel.as_str())));
    }
    output.push_str(")\n\nCORPUS_FAMILY_NAMES: tuple[str, ...] = (\n");
    for family in CorpusFamily::ALL {
        output.push_str(&format!("    {},\n", python_string(family.as_str())));
    }
    output.push_str(")\n\nCODE_LANGUAGE_NAMES: tuple[str, ...] = (\n");
    for language in CodeLanguage::ALL {
        output.push_str(&format!("    {},\n", python_string(language.as_str())));
    }
    output.push_str(")\n");
    for (constant, kind) in [
        ("BENCHMARK_SCALAR_TARGETS", BenchmarkTargetKind::Scalar),
        ("BENCHMARK_TOKEN_TARGETS", BenchmarkTargetKind::Token),
        (
            "BENCHMARK_REGRESSION_TARGETS",
            BenchmarkTargetKind::Regression,
        ),
    ] {
        output.push_str(&format!("\n{constant}: tuple[str, ...] = (\n"));
        for target in BenchmarkProxyTarget::ALL
            .iter()
            .filter(|target| target.kind() == kind)
        {
            output.push_str(&format!("    {},\n", python_string(target.as_str())));
        }
        output.push_str(")\n");
    }
    for (constant, values) in [
        (
            "TRAINING_SCOPE_GROUP_NAMES",
            TrainingGroup::ALL
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "ARCHITECTURE_AXIS_NAMES",
            ArchitectureAxis::ALL
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "JSON_ACTION_NAMES",
            JsonAction::ALL
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "TREE_DOMAIN_NAMES",
            TreeDomain::ALL
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            "OUTPUT_MODE_NAMES",
            OutputMode::ALL
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
        ),
    ] {
        output.push_str(&format!("\n{constant}: tuple[str, ...] = (\n"));
        for value in values {
            output.push_str(&format!("    {},\n", python_string(value)));
        }
        output.push_str(")\n");
    }
    output.push_str("\nOUTPUT_CONTRACT_ROWS: tuple[tuple[str, bool, bool, bool], ...] = (\n");
    for contract in StructuredOutputProfile::CONTRACTS {
        output.push_str(&format!(
            "    ({}, {}, {}, {}),\n",
            python_string(contract.name),
            python_bool(contract.structural),
            python_bool(contract.schema),
            python_bool(contract.executable),
        ));
    }
    output.push_str(")\n");
    output
}

fn parse_args() -> Result<(OutputFormat, Option<String>, Option<String>), String> {
    let mut format = OutputFormat::Json;
    let mut output = None;
    let mut check = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--format" => {
                format = match args.next().as_deref() {
                    Some("json") => OutputFormat::Json,
                    Some("python") => OutputFormat::Python,
                    Some(value) => return Err(format!("unsupported format: {value}")),
                    None => return Err("--format requires a value".to_owned()),
                };
            }
            "--output" => {
                output = Some(args.next().ok_or("--output requires a path")?);
            }
            "--check" => {
                check = Some(args.next().ok_or("--check requires a path")?);
            }
            "--help" | "-h" => {
                return Err(
                    "usage: nlp_flow_codegen [--format json|python] [--output PATH] [--check PATH]"
                        .to_owned(),
                );
            }
            value => return Err(format!("unknown argument: {value}")),
        }
    }
    if output.is_some() && check.is_some() {
        return Err("--output and --check are mutually exclusive".to_owned());
    }
    Ok((format, output, check))
}

fn run() -> Result<(), String> {
    let (format, output, check) = parse_args()?;
    let rendered = match format {
        OutputFormat::Json => render_json(),
        OutputFormat::Python => render_python(),
    };
    if let Some(path) = check {
        let current =
            fs::read_to_string(&path).map_err(|error| format!("cannot read {path}: {error}"))?;
        if current != rendered {
            return Err(format!(
                "{path} is stale; regenerate it with rust/nlp_flow_codegen"
            ));
        }
        return Ok(());
    }
    if let Some(path) = output {
        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        fs::write(&path, rendered).map_err(|error| format!("cannot write {path}: {error}"))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_channels_are_ordered_and_unique() {
        let names: Vec<_> = TextChannel::ALL
            .iter()
            .map(|channel| channel.as_str())
            .collect();
        assert_eq!(
            names,
            ["syntax", "symbol", "discourse", "execution", "engineering"]
        );
        assert_eq!(
            TextChannel::from_name("execution").map(TextChannel::as_str),
            Some("execution")
        );
        assert_eq!(
            CorpusFamily::ALL
                .iter()
                .map(|family| family.as_str())
                .collect::<Vec<_>>(),
            [
                "general_language",
                "code",
                "model_architecture",
                "engineering_acceleration"
            ]
        );
        assert_eq!(
            CodeLanguage::ALL
                .iter()
                .map(|language| language.as_str())
                .collect::<Vec<_>>(),
            ["python", "rust", "cpp"]
        );
    }

    #[test]
    fn hierarchy_exports_the_expected_state_levels() {
        let ids: Vec<_> = NlpHierarchy::SPECS.iter().map(|spec| spec.id).collect();
        assert_eq!(ids, ["L0", "L1", "L2", "L3"]);
        assert_eq!(
            NlpHierarchy::from_id("L2").map(NlpHierarchy::level_id),
            Some("L2")
        );
    }

    #[test]
    fn python_render_is_import_shaped() {
        let rendered = render_python();
        assert!(rendered.contains("HIERARCHY_ROWS"));
        assert!(rendered.contains("\"token_span_surface\""));
        assert!(rendered.contains("CORPUS_FAMILY_NAMES"));
        assert!(rendered.contains("CODE_LANGUAGE_NAMES"));
        assert!(rendered.contains("BENCHMARK_SCALAR_TARGETS"));
        assert!(rendered.contains("ARCHITECTURE_AXIS_NAMES"));
        assert!(rendered.contains("JSON_ACTION_NAMES"));
        assert!(rendered.contains("OUTPUT_CONTRACT_ROWS"));
        assert!(rendered.ends_with(")\n"));
    }

    #[test]
    fn benchmark_targets_have_explicit_and_unique_families() {
        let scalar = BenchmarkProxyTarget::ALL
            .iter()
            .filter(|target| target.kind() == BenchmarkTargetKind::Scalar)
            .count();
        let token = BenchmarkProxyTarget::ALL
            .iter()
            .filter(|target| target.kind() == BenchmarkTargetKind::Token)
            .count();
        let regression = BenchmarkProxyTarget::ALL
            .iter()
            .filter(|target| target.kind() == BenchmarkTargetKind::Regression)
            .count();
        assert_eq!((scalar, token, regression), (27, 8, 3));
    }

    #[test]
    fn json_contract_does_not_require_execution() {
        let contract = StructuredOutputProfile::Json.contract();
        assert!(contract.structural);
        assert!(contract.schema);
        assert!(!contract.executable);
    }
}
