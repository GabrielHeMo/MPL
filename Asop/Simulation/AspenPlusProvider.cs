using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Asop.Simulation
{
    /// <summary>
    /// Owns one Aspen Plus COM instance on a dedicated STA thread.
    /// Every COM call is serialized onto that same thread.
    /// </summary>
    public sealed class AspenPlusProvider : ISimulationProvider
    {
        private const string RunStatusPath = @"\Data\Results Summary\Run-Status\Output\UOSSTAT2";

        private readonly StaComWorker _worker = new StaComWorker("MPL-AspenPlus-STA");
        private dynamic? _aspen;
        private volatile bool _isOpen;
        private bool _disposed;

        public bool IsOpen => _isOpen;

        public Task OpenAsync(string filePath, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Aspen file path is empty.", nameof(filePath));

            return _worker.InvokeAsync(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (_aspen != null)
                    CloseInternal();

                Type? aspenType = Type.GetTypeFromProgID("Apwn.Document");
                if (aspenType == null)
                    throw new InvalidOperationException("Aspen Plus COM ProgID 'Apwn.Document' was not found.");

                _aspen = Activator.CreateInstance(aspenType)
                    ?? throw new InvalidOperationException("Could not create the Aspen Plus COM instance.");

                _aspen.InitFromArchive2(filePath);
                _aspen.Visible = false;

                try
                {
                    _aspen.SuppressDialogs = true;
                }
                catch
                {
                    // Not all Aspen versions expose this property.
                }

                _isOpen = true;
            });
        }

        public Task SetValueAsync(string path, double value, CancellationToken cancellationToken = default)
        {
            return _worker.InvokeAsync(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                dynamic aspen = RequireAspen();
                dynamic node = aspen.Tree.FindNode(path);

                if (node == null)
                    throw new InvalidOperationException($"Aspen variable was not found: {path}");

                node.Value = value;
            });
        }

        public Task<double> GetValueAsync(string path, CancellationToken cancellationToken = default)
        {
            return _worker.InvokeAsync(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                dynamic aspen = RequireAspen();
                dynamic node = aspen.Tree.FindNode(path);

                if (node == null)
                    throw new InvalidOperationException($"Aspen output was not found: {path}");

                object? raw = node.Value;
                if (raw == null)
                    throw new InvalidOperationException($"Aspen output has no value: {path}");

                double value = Convert.ToDouble(raw, CultureInfo.InvariantCulture);
                if (double.IsNaN(value) || double.IsInfinity(value))
                    throw new InvalidOperationException($"Aspen output is not finite: {path} = {value}");

                return value;
            });
        }

        public Task RunAsync(CancellationToken cancellationToken = default)
        {
            return _worker.InvokeAsync(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                dynamic aspen = RequireAspen();

                aspen.Run2();

                while (aspen.Engine.IsRunning)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Thread.Sleep(200);
                }
            });
        }

        public Task ReinitializeAsync(CancellationToken cancellationToken = default)
        {
            return _worker.InvokeAsync(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                dynamic aspen = RequireAspen();

                try
                {
                    aspen.Reinit();
                }
                catch
                {
                    // Keep the original behavior: reinitialization is best effort.
                }
            });
        }

        public Task<int?> GetRunStatusAsync(CancellationToken cancellationToken = default)
        {
            return _worker.InvokeAsync<int?>(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                dynamic aspen = RequireAspen();
                dynamic node = aspen.Tree.FindNode(RunStatusPath);

                if (node == null || node.Value == null)
                    return null;

                return Convert.ToInt32(node.Value, CultureInfo.InvariantCulture);
            });
        }

        public Task CloseAsync(CancellationToken cancellationToken = default)
        {
            return _worker.InvokeAsync(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                CloseInternal();
            });
        }

        private dynamic RequireAspen()
        {
            if (_aspen == null)
                throw new InvalidOperationException("Aspen Plus is not open.");

            return _aspen;
        }

        private void CloseInternal()
        {
            if (_aspen == null)
            {
                _isOpen = false;
                return;
            }

            try
            {
                _aspen.Close(false);
            }
            catch
            {
            }

            try
            {
                if (Marshal.IsComObject(_aspen))
                    Marshal.FinalReleaseComObject(_aspen);
            }
            catch
            {
            }

            _aspen = null;
            _isOpen = false;
        }

        public async ValueTask DisposeAsync()
        {
            if (_disposed)
                return;

            _disposed = true;

            try
            {
                await CloseAsync();
            }
            catch
            {
            }

            _worker.Dispose();
        }

        /// <summary>
        /// Minimal single-threaded apartment executor for COM automation.
        /// </summary>
        private sealed class StaComWorker : IDisposable
        {
            private readonly BlockingCollection<Action> _queue = new BlockingCollection<Action>();
            private readonly Thread _thread;
            private bool _disposed;

            public StaComWorker(string name)
            {
                _thread = new Thread(Run)
                {
                    IsBackground = true,
                    Name = name
                };
                _thread.SetApartmentState(ApartmentState.STA);
                _thread.Start();
            }

            public Task InvokeAsync(Action action)
            {
                return InvokeAsync<object?>(() =>
                {
                    action();
                    return null;
                });
            }

            public Task<T> InvokeAsync<T>(Func<T> func)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(StaComWorker));

                TaskCompletionSource<T> tcs = new TaskCompletionSource<T>(
                    TaskCreationOptions.RunContinuationsAsynchronously);

                _queue.Add(() =>
                {
                    try
                    {
                        tcs.SetResult(func());
                    }
                    catch (Exception ex)
                    {
                        tcs.SetException(ex);
                    }
                });

                return tcs.Task;
            }

            private void Run()
            {
                foreach (Action action in _queue.GetConsumingEnumerable())
                    action();
            }

            public void Dispose()
            {
                if (_disposed)
                    return;

                _disposed = true;
                _queue.CompleteAdding();

                if (Thread.CurrentThread != _thread)
                    _thread.Join();

                _queue.Dispose();
            }
        }
    }
}
