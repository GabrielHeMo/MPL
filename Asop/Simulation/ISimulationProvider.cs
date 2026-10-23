using System;
using System.Threading;
using System.Threading.Tasks;

namespace Asop.Simulation
{
    /// <summary>
    /// Abstraction for a process simulator used by MPL.
    /// Python never receives or owns the simulator COM object.
    /// </summary>
    public interface ISimulationProvider : IAsyncDisposable
    {
        bool IsOpen { get; }

        Task OpenAsync(string filePath, CancellationToken cancellationToken = default);
        Task SetValueAsync(string path, double value, CancellationToken cancellationToken = default);
        Task<double> GetValueAsync(string path, CancellationToken cancellationToken = default);
        Task RunAsync(CancellationToken cancellationToken = default);
        Task ReinitializeAsync(CancellationToken cancellationToken = default);
        Task<int?> GetRunStatusAsync(CancellationToken cancellationToken = default);
        Task CloseAsync(CancellationToken cancellationToken = default);
    }
}
