using System;
using System.Collections.Generic;
using System.Text;

using System;

namespace Asop.Models
{
    public sealed class ActivityLogEntry
    {
        public DateTime Timestamp { get; set; } = DateTime.Now;
        public string Level { get; set; } = "INFO";
        public string Message { get; set; } = "";

        public string TimestampText => Timestamp.ToString("HH:mm:ss");
    }
}
