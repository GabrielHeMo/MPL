using Asop.Simulation;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;

namespace Asop.Evaluation
{
    /// <summary>
    /// Executes physical Aspen evaluations. Objective-function mathematics stays in
    /// Python during this first migration step; Aspen COM stays exclusively in C#.
    /// </summary>
    public sealed class AspenEvaluationEngine
    {
        private const int FailedStatus1 = 9;
        private const int FailedStatus2 = 10;

        private readonly ISimulationProvider _provider;
        private readonly OptimizationBridgeConfig _config;

        public AspenEvaluationEngine(
            ISimulationProvider provider,
            OptimizationBridgeConfig config)
        {
            _provider = provider ?? throw new ArgumentNullException(nameof(provider));
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        public async Task<SimulationEvaluationResult> EvaluateAsync(
            double[] x,
            CancellationToken cancellationToken = default)
        {
            double[] xProcessed = PreprocessX(x);

            try
            {
                // Preserve the restart behavior currently implemented in aspenproblems.py.
                for (int i = 0; i < _config.Inputs.Count; i++)
                {
                    OptimizationInputDefinition input = _config.Inputs[i];
                    double restart = ParseRestartValue(input);
                    await _provider.SetValueAsync(input.Path, restart, cancellationToken).ConfigureAwait(false);
                }

                // Apply candidate suggested by Python/pymoo.
                for (int i = 0; i < _config.Inputs.Count; i++)
                {
                    await _provider.SetValueAsync(
                        _config.Inputs[i].Path,
                        xProcessed[i],
                        cancellationToken).ConfigureAwait(false);
                }

                await _provider.RunAsync(cancellationToken).ConfigureAwait(false);

                int? status = await _provider.GetRunStatusAsync(cancellationToken).ConfigureAwait(false);
                if (!status.HasValue)
                {
                    await SafeReinitializeAsync(cancellationToken).ConfigureAwait(false);
                    return Failure(xProcessed, null, "Aspen Run-Status could not be read.");
                }

                if (status.Value == FailedStatus1 || status.Value == FailedStatus2)
                {
                    await SafeReinitializeAsync(cancellationToken).ConfigureAwait(false);
                    return Failure(
                        xProcessed,
                        status,
                        $"Aspen did not converge. Run-Status = {status.Value}");
                }

                Dictionary<string, double?> outputs = new();
                foreach (OptimizationOutputDefinition output in _config.Outputs)
                {
                    double value = await _provider.GetValueAsync(output.Path, cancellationToken).ConfigureAwait(false);
                    outputs[output.Tag] = value;
                }

                return new SimulationEvaluationResult
                {
                    Inputs = xProcessed,
                    Outputs = outputs,
                    Success = true,
                    AspenConverged = true,
                    RunStatus = status,
                    Message = $"Aspen converged. Run-Status = {status.Value}"
                };
            }
            catch (Exception ex)
            {
                await SafeReinitializeAsync(cancellationToken).ConfigureAwait(false);
                return Failure(xProcessed, null, $"Error evaluating point in Aspen: {ex.Message}");
            }
        }

        public async Task<IReadOnlyList<SimulationEvaluationResult>> EvaluateBatchAsync(
            IReadOnlyList<double[]> points,
            IProgress<EvaluationProgress>? progress = null,
            CancellationToken cancellationToken = default)
        {
            List<SimulationEvaluationResult> results = new(points.Count);
            int success = 0;
            int failed = 0;

            for (int i = 0; i < points.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                SimulationEvaluationResult result = await EvaluateAsync(
                    points[i],
                    cancellationToken).ConfigureAwait(false);

                results.Add(result);

                if (result.Success)
                    success++;
                else
                    failed++;

                progress?.Report(new EvaluationProgress
                {
                    Completed = i + 1,
                    Total = points.Count,
                    Successful = success,
                    Failed = failed
                });
            }

            return results;
        }

        private double[] PreprocessX(double[] x)
        {
            if (x.Length != _config.Inputs.Count)
            {
                throw new ArgumentException(
                    $"Expected {_config.Inputs.Count} decision variables but received {x.Length}.",
                    nameof(x));
            }

            double[] processed = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                double value = x[i];
                string type = _config.Inputs[i].VariableType ?? "Continuous";

                if (type.Equals("Discrete", StringComparison.OrdinalIgnoreCase) ||
                    type.Equals("Integer", StringComparison.OrdinalIgnoreCase))
                {
                    value = Math.Round(value);
                }

                processed[i] = value;
            }

            return processed;
        }

        private static double ParseRestartValue(OptimizationInputDefinition input)
        {
            string text = string.IsNullOrWhiteSpace(input.RestartValue)
                ? input.CurrentValue
                : input.RestartValue;

            if (string.IsNullOrWhiteSpace(text))
                throw new InvalidOperationException($"No restart value exists for {input.Tag} ({input.Path}).");

            text = text.Replace(',', '.');

            if (!double.TryParse(
                text,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out double value))
            {
                throw new InvalidOperationException(
                    $"Invalid restart value for {input.Tag}: '{text}'.");
            }

            if (input.VariableType.Equals("Discrete", StringComparison.OrdinalIgnoreCase) ||
                input.VariableType.Equals("Integer", StringComparison.OrdinalIgnoreCase))
            {
                value = Math.Round(value);
            }

            return value;
        }

        private async Task SafeReinitializeAsync(CancellationToken cancellationToken)
        {
            try
            {
                await _provider.ReinitializeAsync(cancellationToken).ConfigureAwait(false);
            }
            catch
            {
            }
        }

        private SimulationEvaluationResult Failure(double[] x, int? status, string message)
        {
            Dictionary<string, double?> outputs = new();
            foreach (OptimizationOutputDefinition output in _config.Outputs)
                outputs[output.Tag] = null;

            return new SimulationEvaluationResult
            {
                Inputs = x,
                Outputs = outputs,
                Success = false,
                AspenConverged = false,
                RunStatus = status,
                Message = message
            };
        }
    }
}
