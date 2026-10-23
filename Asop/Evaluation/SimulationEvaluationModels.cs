using Newtonsoft.Json;
using System.Collections.Generic;

namespace Asop.Evaluation
{
    public sealed class OptimizationBridgeConfig
    {
        [JsonProperty("inputs")]
        public List<OptimizationInputDefinition> Inputs { get; set; } = new();

        [JsonProperty("outputs")]
        public List<OptimizationOutputDefinition> Outputs { get; set; } = new();
    }

    public sealed class OptimizationInputDefinition
    {
        [JsonProperty("tag")]
        public string Tag { get; set; } = "";

        [JsonProperty("path")]
        public string Path { get; set; } = "";

        [JsonProperty("current_value")]
        public string CurrentValue { get; set; } = "";

        [JsonProperty("restart_values")]
        public string RestartValue { get; set; } = "";

        [JsonProperty("variable_type")]
        public string VariableType { get; set; } = "Continuous";
    }

    public sealed class OptimizationOutputDefinition
    {
        [JsonProperty("tag")]
        public string Tag { get; set; } = "";

        [JsonProperty("path")]
        public string Path { get; set; } = "";
    }

    public sealed class SimulationEvaluationResult
    {
        public double[] Inputs { get; set; } = System.Array.Empty<double>();
        public Dictionary<string, double?> Outputs { get; set; } = new();
        public bool Success { get; set; }
        public bool AspenConverged { get; set; }
        public int? RunStatus { get; set; }
        public string Message { get; set; } = "";
    }

    public sealed class EvaluationProgress
    {
        public int Completed { get; init; }
        public int Total { get; init; }
        public int Successful { get; init; }
        public int Failed { get; init; }
    }
}
