using Asop.Models;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Windows;


namespace Asop.Aspen
{
    public class AspenConnector
    {
        private dynamic? aspenApp;

        public dynamic? AspenApp
        {
            get { return aspenApp; }
        }

        public bool OpenSimulation(string filePath)
        {
            try
            {
                // Crear instancia Aspen
                Type? aspenType = Type.GetTypeFromProgID("Apwn.Document");

                if (aspenType == null)
                {
                    return false;
                }

                aspenApp = Activator.CreateInstance(aspenType);

                if (aspenApp == null)
                    return false;

                // Abrir archivo .bkp
                aspenApp.InitFromArchive2(filePath);

                // Mostrar Aspen
                aspenApp.Visible = false;

                return true;
            }
            catch
            {
                return false;
            }
        }




        public List<AspenVariable> ReadTree()
        {
            List<AspenVariable> variables = new List<AspenVariable>();

            if (aspenApp == null)
                return variables;

            ReadBranch(@"\Data\Streams", variables);
            ReadBranch(@"\Data\Blocks", variables);

            return variables;
        }


        private void ReadBranch(string branchPath, List<AspenVariable> variables)
        {
            if (aspenApp == null)
                return;

            try
            {
                dynamic branch = aspenApp.Tree.FindNode(branchPath);

                if (branch == null)
                    return;

                TraverseNode(branch, branchPath, variables, 0);
            }
            catch
            {
            }
        }



        private void TraverseNode(dynamic node, string currentPath, List<AspenVariable> variables, int depth)
        {
            if (depth > 4)
                return;

            try
            {
                string value = SafeGetValue(node);

                if (IsNumericValue(value))
                {
                    variables.Add(new AspenVariable
                    {
                        Path = currentPath,
                        Value = value,
                        Type = GuessType(currentPath),
                        Editable = IsEditable(currentPath)
                    });
                }
            }
            catch
            {
            }

            try
            {
                foreach (dynamic child in node.Elements)
                {
                    string? childName = "";

                    try
                    {
                        childName = Convert.ToString(child.Name);
                    }
                    catch
                    {
                        continue;
                    }

                    string childPath = currentPath + @"\" + childName;

                    TraverseNode(child, childPath, variables, depth + 1);
                }
            }
            catch
            {
            }
        }



        private string SafeGetValue(dynamic node)
        {
            try
            {
                object? value = node.Value;

                if (value == null)
                    return "";

                return Convert.ToString(value, CultureInfo.InvariantCulture) ?? "";
            }
            catch
            {
                return "";
            }
        }

        private string GuessType(string path)
        {
            //path = path.ToLower();
            string lowerPath = path.ToLowerInvariant();

            if (lowerPath.Contains("input"))
                return "Input";

            if (lowerPath.Contains("output"))
                return "Output";

            if (lowerPath.Contains("block"))
                return "Block";

            if (lowerPath.Contains("stream"))
                return "Stream";

            return "Unknown";
        }

        private static bool IsEditable(string path)
        {
            string lowerPath = path.ToLowerInvariant();

            if (lowerPath.Contains("output"))
                return false;

            if (lowerPath.Contains("input"))
                return true;

            return false;
        }

        private static bool IsNumericValue(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return false;

            return double.TryParse(
                value,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out _
            );
        }


        public void CloseSimulation()
        {
            try
            {
                if (aspenApp != null)
                {
                    aspenApp.Close(false);

                    System.Runtime.InteropServices.Marshal.ReleaseComObject(aspenApp);

                    aspenApp = null;
                }
            }
            catch
            {
                aspenApp = null;
            }
        }

        public bool RunSimulation()
        {
            if (aspenApp == null)
                return false;

            try
            {
                aspenApp.Run2();

                while (aspenApp.Engine.IsRunning)
                {
                    System.Threading.Thread.Sleep(200);
                }

                return true;
            }
            catch
            {
                return false;
            }
        }



    }
}
