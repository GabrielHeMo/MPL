using Asop.Evaluation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Asop.PythonIntegration
{
    /// <summary>
    /// Interactive JSONL bridge:
    /// Python -> stdout: __MPL__{evaluate_batch request | status message}
    /// C#     -> stdin : {batch_result response}\n
    /// All non-prefixed Python stdout is treated as diagnostic text.
    /// </summary>
    public sealed class OptimizerStatusMessage
    {
        public string Level { get; init; } = "INFO";
        public string Message { get; init; } = "";
        public double? Progress { get; init; }
        public string? Stage { get; init; }
        public int? Current { get; init; }
        public int? Total { get; init; }
        public string? Algorithm { get; init; }
        public bool Log { get; init; } = true;
    }

    public static class PythonJsonlOptimizerBridge
    {
        public const string ProtocolPrefix = "__MPL__";

        public static async Task<string> RunAsync(
            string pythonExe,
            string scriptPath,
            string configPath,
            string resultsPath,
            AspenEvaluationEngine engine,
            IProgress<EvaluationProgress>? progress = null,
            CancellationToken cancellationToken = default,
            IProgress<OptimizerStatusMessage>? statusProgress = null)
        {
            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" \"{configPath}\" \"{resultsPath}\"",
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using Process process = new Process
            {
                StartInfo = start,
                EnableRaisingEvents = true
            };

            StringBuilder stdoutLog = new StringBuilder();
            process.Start();
            process.StandardInput.AutoFlush = true;

            Task<string> stderrTask = process.StandardError.ReadToEndAsync();

            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();

                string? line = await process.StandardOutput.ReadLineAsync().ConfigureAwait(false);
                if (line == null)
                    break;

                if (!line.StartsWith(ProtocolPrefix, StringComparison.Ordinal))
                {
                    stdoutLog.AppendLine(line);
                    continue;
                }

                string payload = line.Substring(ProtocolPrefix.Length);
                JObject request = JObject.Parse(payload);
                string type = request.Value<string>("type") ?? "";

                if (type.Equals("status", StringComparison.OrdinalIgnoreCase))
                {
                    statusProgress?.Report(new OptimizerStatusMessage
                    {
                        Level = request.Value<string>("level") ?? "INFO",
                        Message = request.Value<string>("message") ?? "",
                        Progress = request.Value<double?>("progress"),
                        Stage = request.Value<string>("stage"),
                        Current = request.Value<int?>("current"),
                        Total = request.Value<int?>("total"),
                        Algorithm = request.Value<string>("algorithm"),
                        Log = request.Value<bool?>("log") ?? true
                    });
                    continue;
                }

                if (!type.Equals("evaluate_batch", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidOperationException(
                        $"Unknown Python bridge request type: '{type}'.");
                }

                string requestId = request.Value<string>("request_id") ?? Guid.NewGuid().ToString("N");
                JArray? xArray = request["X"] as JArray;

                if (xArray == null)
                    throw new InvalidOperationException("evaluate_batch request does not contain X.");

                List<double[]> points = xArray.ToObject<List<double[]>>() ?? new List<double[]>();

                IReadOnlyList<SimulationEvaluationResult> evaluations =
                    await engine.EvaluateBatchAsync(points, progress, cancellationToken).ConfigureAwait(false);

                var response = new
                {
                    type = "batch_result",
                    request_id = requestId,
                    results = evaluations.Select(r => new
                    {
                        success = r.Success,
                        x = r.Inputs,
                        outputs = r.Outputs,
                        aspen_converged = r.AspenConverged,
                        run_status = r.RunStatus,
                        message = r.Message
                    }).ToList()
                };

                string responseJson = JsonConvert.SerializeObject(response, Formatting.None);
                await process.StandardInput.WriteLineAsync(responseJson).ConfigureAwait(false);
                await process.StandardInput.FlushAsync().ConfigureAwait(false);
            }

            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            string stderr = await stderrTask.ConfigureAwait(false);

            if (process.ExitCode != 0)
            {
                throw new Exception(
                    "Python ended with an error.\n\n" +
                    "STDOUT:\n" + stdoutLog + "\n\n" +
                    "STDERR:\n" + stderr);
            }

            if (!File.Exists(resultsPath))
            {
                throw new Exception(
                    "Python finished successfully but did not generate the results file.\n\n" +
                    "STDOUT:\n" + stdoutLog + "\n\n" +
                    "STDERR:\n" + stderr);
            }

            return await File.ReadAllTextAsync(resultsPath, cancellationToken).ConfigureAwait(false);
        }
    }
}
