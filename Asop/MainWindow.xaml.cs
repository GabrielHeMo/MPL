using Asop.Aspen;
using Asop.Models;
using Asop.Evaluation;
using Asop.PythonIntegration;
using Asop.Simulation;
using Microsoft.Win32;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Printing;
using System.Security.Policy;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using static System.Runtime.InteropServices.JavaScript.JSType;


namespace Asop
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    /// private string selectedAspenFile = "";
    /// 


    public partial class MainWindow : Window
    {
        private string selectedAspenFile = "";
        private string lastResultsPath = "";
        private string lastResultsJson = "";

        public string OutputFile { get; set; } = "optimization_results";

        private AspenConnector connector = new AspenConnector();

        private List<AspenVariable> allVariables = new List<AspenVariable>();
        private AspenVariable? selectedVariable = null;

        private ObservableCollection<AspenVariable> inputParameters =
            new ObservableCollection<AspenVariable>();

        private ObservableCollection<AspenVariable> outputObjectives =
            new ObservableCollection<AspenVariable>();

        private ObservableCollection<ObjectiveFunction> objectiveFunctions =
            new ObservableCollection<ObjectiveFunction>();

        private ObservableCollection<OptimizationConstraint> optimizationConstraints =
            new ObservableCollection<OptimizationConstraint>();

        public ObservableCollection<string> InputTags { get; set; } = new ObservableCollection<string>();


        private Dictionary<string, TextBox> hyperparameterTextBoxes =
            new Dictionary<string, TextBox>();

        private Dictionary<string, ComboBox> hyperparameterComboBoxes =
            new Dictionary<string, ComboBox>();

        public ObservableCollection<ActivityLogEntry> ActivityLog { get; } =
            new ObservableCollection<ActivityLogEntry>();

        private const int MaxActivityLogEntries = 2000;
        private int lastLoggedBatchProgress = -1;

        public MainWindow()
        {
            // ESTE ES EL CONSTRUCTOR DE LA CLASE, SE EJECUTA CUANDO SE CREA UNA INSTANCIA DE MainWindow
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            Thread.CurrentThread.CurrentUICulture = CultureInfo.InvariantCulture;


            this.WindowStartupLocation = WindowStartupLocation.CenterScreen;
            this.WindowState = WindowState.Maximized;

            InitializeComponent();
            DataContext = this;
            InputParametersGrid.ItemsSource = inputParameters;
            OutputObjectivesGrid.ItemsSource = outputObjectives;
            ObjectiveFunctionsGrid.ItemsSource = objectiveFunctions;
            ConstraintsGrid.ItemsSource = optimizationConstraints;
            LeftVariableColumn.ItemsSource = InputTags;

            SetProgress(0);
            SetActivityStatus("Ready.");
            AddLog("INFO", "MPL Optimizer initialized.");
        }


        // Centralized activity log used by UI actions and optimization backends.
        private void AddLog(string level, string message)
        {
            if (string.IsNullOrWhiteSpace(message))
                return;

            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.BeginInvoke(new Action(() => AddLog(level, message)));
                return;
            }

            ActivityLogEntry entry = new ActivityLogEntry
            {
                Timestamp = DateTime.Now,
                Level = string.IsNullOrWhiteSpace(level) ? "INFO" : level.ToUpperInvariant(),
                Message = message.Trim()
            };

            ActivityLog.Add(entry);

            while (ActivityLog.Count > MaxActivityLogEntries)
            {
                ActivityLog.RemoveAt(0);
            }

            ActivityLogListBox?.ScrollIntoView(entry);
        }

        private void SetActivityStatus(string status)
        {
            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.BeginInvoke(new Action(() => SetActivityStatus(status)));
                return;
            }

            TxtActivityStatus.Text = status;
        }

        private void SetProgress(double? percent, bool indeterminate = false)
        {
            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.BeginInvoke(new Action(() => SetProgress(percent, indeterminate)));
                return;
            }

            OptimizationProgressBar.IsIndeterminate = indeterminate;

            if (indeterminate || !percent.HasValue)
            {
                OptimizationProgressBar.Value = 0;
                TxtProgressPercent.Text = indeterminate ? "Working..." : "0%";
                return;
            }

            double safePercent = Math.Max(0.0, Math.Min(100.0, percent.Value));
            OptimizationProgressBar.Value = safePercent;
            TxtProgressPercent.Text = $"{safePercent:0}%";
        }

        private void BtnClearLog_Click(object sender, RoutedEventArgs e)
        {
            ActivityLog.Clear();
        }

        private void HandleOptimizerStatus(OptimizerStatusMessage status)
        {
            if (status == null || string.IsNullOrWhiteSpace(status.Message))
                return;

            string level = string.IsNullOrWhiteSpace(status.Level)
                ? "INFO"
                : status.Level.ToUpperInvariant();

            if (status.Log)
            {
                AddLog(level, status.Message);
            }

            // TxtOptimizationStatus.Text = status.Message;
            SetActivityStatus(status.Message);

            if (status.Progress.HasValue)
            {
                SetProgress(status.Progress.Value);
            }
        }

        private bool TryHandleOptimizerStatusLine(string line)
        {
            if (string.IsNullOrWhiteSpace(line) ||
                !line.StartsWith(PythonJsonlOptimizerBridge.ProtocolPrefix, StringComparison.Ordinal))
            {
                return false;
            }

            try
            {
                string payload = line.Substring(PythonJsonlOptimizerBridge.ProtocolPrefix.Length);
                JObject message = JObject.Parse(payload);
                string type = message.Value<string>("type") ?? "";

                if (!type.Equals("status", StringComparison.OrdinalIgnoreCase))
                    return false;

                HandleOptimizerStatus(new OptimizerStatusMessage
                {
                    Level = message.Value<string>("level") ?? "INFO",
                    Message = message.Value<string>("message") ?? "",
                    Progress = message.Value<double?>("progress"),
                    Stage = message.Value<string>("stage"),
                    Current = message.Value<int?>("current"),
                    Total = message.Value<int?>("total"),
                    Algorithm = message.Value<string>("algorithm"),
                    Log = message.Value<bool?>("log") ?? true
                });

                return true;
            }
            catch (Exception ex)
            {
                AddLog("WARNING", $"Could not parse an MPL optimizer status message: {ex.Message}");
                return false;
            }
        }

        private async Task PumpProcessStreamAsync(
            StreamReader reader,
            StringBuilder buffer,
            string level)
        {
            string? line;

            while ((line = await reader.ReadLineAsync()) != null)
            {
                buffer.AppendLine(line);

                if (!string.IsNullOrWhiteSpace(line) && !TryHandleOptimizerStatusLine(line))
                {
                    AddLog(level, line);
                }
            }
        }


        // Problema de Json file
        private static string GetMplDataDirectory()
        {
            string localAppData = Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData
            );

            string mplDirectory = Path.Combine(
                localAppData,
                "MPL Optimizer"
            );

            Directory.CreateDirectory(mplDirectory);

            return mplDirectory;
        }

        // BOTON QUE PERMITE ENCONTRAR RUTA DE ASPEN
        private void BtnFind_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog
            {
                Title = "Select an Aspen file",
                Filter = "Aspen Backup Files (*.bkp)|*.bkp"
            };

            if (dialog.ShowDialog() == true)
            {
                selectedAspenFile = dialog.FileName;
                TxtFilePath.Text = selectedAspenFile;

                AddLog("SUCCESS", $"Aspen file selected: {Path.GetFileName(selectedAspenFile)}");
                SetActivityStatus("Aspen file selected.");
            }
        }


        // BOTON QUE PERMITE ABRIR RUTA DE ASPEN
        private async void BtnRead_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(selectedAspenFile))
            {
                AddLog("WARNING", "Select an Aspen file before reading the Aspen tree.");
                SetActivityStatus("Aspen file required.");
                return;
            }

            BtnRead.IsEnabled = false;
            SetActivityStatus("Opening Aspen Plus...");
            SetProgress(null, true);
            AddLog("INFO", "Opening Aspen Plus and preparing to read the simulation tree...");

            try
            {
                bool success = connector.OpenSimulation(selectedAspenFile);

                if (!success)
                {
                    AddLog("ERROR", "Aspen Plus could not be opened.");
                    SetActivityStatus("Could not open Aspen Plus.");
                    return;
                }

                AddLog("SUCCESS", "Aspen Plus simulation opened successfully.");
                AddLog("INFO", "Reading numeric variables from the Aspen tree...");

                await Task.Delay(3000); // allows Aspen to finish loading

                List<AspenVariable> variables = await Task.Run(() => connector.ReadTree());

                allVariables = variables;
                BuildTreeView(allVariables);

                AddLog("SUCCESS", $"Aspen tree loaded. {allVariables.Count} numeric variables were found.");
                SetActivityStatus("Aspen tree ready.");
                SetProgress(100);
            }
            catch (Exception ex)
            {
                AddLog("ERROR", $"Error while reading the Aspen tree: {ex.Message}");
                SetActivityStatus("Aspen tree read failed.");
                SetProgress(0);
            }
            finally
            {
                BtnRead.IsEnabled = true;
            }
        }

        // BOTON QUE LIMPIA TODO HASTA EL MOMENTO ENCONTRAR RUTA DE ASPEN
        private void BtnClean_Click(object sender, RoutedEventArgs e)
        {
            connector.CloseSimulation();

            AspenTreeView.Items.Clear();
            inputParameters.Clear();
            outputObjectives.Clear();
            objectiveFunctions.Clear();
            optimizationConstraints.Clear();
            allVariables.Clear();
            selectedVariable = null;
            selectedAspenFile = "";
            lastResultsPath = "";
            lastResultsJson = "";

            TxtFilePath.Clear();
            BtnExportResults.IsEnabled = false;

            UpdateInputTags();
            UpdateProblemType();
            ValidateOptimizationSetup();

            SetProgress(0);
            SetActivityStatus("Workspace cleaned.");
            AddLog("SUCCESS", "Workspace cleaned and Aspen connection closed.");
        }


        private void BuildTreeView(List<AspenVariable> variables)
        {
            AspenTreeView.Items.Clear();

            Dictionary<string, TreeViewItem> nodes =
                new Dictionary<string, TreeViewItem>();

            foreach (AspenVariable variable in variables)
            {
                string[] parts = variable.Path
                    .Trim('\\')
                    .Split('\\');

                string currentPath = "";

                TreeViewItem? parentItem = null;

                foreach (string part in parts)
                {
                    currentPath += @"\" + part;

                    if (!nodes.ContainsKey(currentPath))
                    {
                        TreeViewItem item = new TreeViewItem();
                        item.Header = part;
                        item.Tag = currentPath;

                        nodes[currentPath] = item;

                        if (parentItem == null)
                        {
                            AspenTreeView.Items.Add(item);
                        }
                        else
                        {
                            parentItem.Items.Add(item);
                        }
                    }

                    parentItem = nodes[currentPath];
                }

                if (parentItem != null)
                {
                    parentItem.Header = $"{parts.Last()} = {variable.Value}";
                    parentItem.Tag = variable;
                }


            }
            // Aspen is only needed while the tree is being read.
            connector.CloseSimulation();
            AddLog("INFO", "Aspen connection closed after reading the tree.");
        }


        private void AspenTreeView_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            TreeViewItem? item = AspenTreeView.SelectedItem as TreeViewItem;

            if (item == null)
            {
                selectedVariable = null;
                return;
            }

            selectedVariable = item.Tag as AspenVariable;
        }

        private void BtnAddSelected_Click(object sender, RoutedEventArgs e)
        {
            if (selectedVariable == null)
            {
                AddLog("WARNING", "Select a numeric variable from the Aspen tree first.");
                return;
            }

            if (selectedVariable.Type == "Input")
            {
                bool alreadyExists = inputParameters.Any(v => v.Path == selectedVariable.Path);

                if (alreadyExists)
                {
                    AddLog("WARNING", $"Input already added: {selectedVariable.Path}");
                    return;
                }

                double currentValue = 0.0;

                double.TryParse(
                    selectedVariable.Value,
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out currentValue
                );

                AspenVariable newInput = new AspenVariable
                {
                    Tag = $"x{inputParameters.Count + 1}",
                    Path = selectedVariable.Path,
                    Value = selectedVariable.Value,
                    Restar = currentValue.ToString(CultureInfo.InvariantCulture),
                    Type = selectedVariable.Type,
                    Editable = selectedVariable.Editable,
                    VariableKind = "Continuous",
                    LowerBound = (currentValue * 0.5).ToString(CultureInfo.InvariantCulture),
                    UpperBound = (currentValue * 1.5).ToString(CultureInfo.InvariantCulture)
                };

                inputParameters.Add(newInput);
                UpdateInputTags();
                ValidateOptimizationSetup();

                AddLog("SUCCESS", $"Added input {newInput.Tag}: {newInput.Path}");
                return;
            }

            if (selectedVariable.Type == "Output")
            {
                bool alreadyExists = outputObjectives.Any(v => v.Path == selectedVariable.Path);

                if (alreadyExists)
                {
                    AddLog("WARNING", $"Output already added: {selectedVariable.Path}");
                    return;
                }

                AspenVariable newOutput = new AspenVariable
                {
                    Tag = $"y{outputObjectives.Count + 1}",
                    Path = selectedVariable.Path,
                    Value = selectedVariable.Value,
                    Type = selectedVariable.Type,
                    Editable = selectedVariable.Editable
                };

                outputObjectives.Add(newOutput);
                ValidateOptimizationSetup();
                AddLog("SUCCESS", $"Added output {newOutput.Tag}: {newOutput.Path}");
                return;
            }

            AddLog("WARNING", "The selected Aspen variable is neither Input nor Output.");
        }


        private void BtnRemoveSelected_Click(object sender, RoutedEventArgs e)
        {
            if (InputParametersGrid.SelectedItem is AspenVariable selectedInput)
            {
                AspenVariable? itemToRemove = inputParameters
                    .FirstOrDefault(v => v.Path == selectedInput.Path);

                if (itemToRemove != null)
                {
                    string removedTag = itemToRemove.Tag;
                    string removedPath = itemToRemove.Path;
                    inputParameters.Remove(itemToRemove);
                    UpdateInputTags();
                    ValidateOptimizationSetup();
                    InputParametersGrid.Items.Refresh();
                    AddLog("INFO", $"Removed input {removedTag}: {removedPath}");
                }

                return;
            }

            if (OutputObjectivesGrid.SelectedItem is AspenVariable selectedOutput)
            {
                AspenVariable? itemToRemove = outputObjectives
                    .FirstOrDefault(v => v.Path == selectedOutput.Path);

                if (itemToRemove != null)
                {
                    string removedTag = itemToRemove.Tag;
                    string removedPath = itemToRemove.Path;
                    outputObjectives.Remove(itemToRemove);
                    OutputObjectivesGrid.Items.Refresh();
                    ValidateOptimizationSetup();
                    AddLog("INFO", $"Removed output {removedTag}: {removedPath}");
                }

                return;
            }

            AddLog("WARNING", "Select an input or output row before removing it.");
        }


        private void InputParametersGrid_CellEditEnding(object sender, DataGridCellEditEndingEventArgs e)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                foreach (AspenVariable variable in inputParameters)
                {
                    string lbText = variable.LowerBound.Replace(",", ".");
                    string ubText = variable.UpperBound.Replace(",", ".");

                    bool lbOk = double.TryParse(
                        lbText,
                        NumberStyles.Float,
                        CultureInfo.InvariantCulture,
                        out double lb
                    );

                    bool ubOk = double.TryParse(
                        ubText,
                        NumberStyles.Float,
                        CultureInfo.InvariantCulture,
                        out double ub
                    );

                    if (!lbOk || !ubOk)
                    {
                        AddLog("WARNING", $"Invalid numeric bounds for {variable.Tag}.");
                        continue;
                    }

                    if (ub <= lb)
                    {
                        AddLog("WARNING", $"Upper Bound must be greater than Lower Bound for {variable.Tag}: {variable.Path}");
                        return;
                    }

                    if (variable.VariableKind == "Discrete" && (lb % 1 != 0 || ub % 1 != 0))
                    {
                        AddLog("WARNING", $"Discrete bounds must be integers for {variable.Tag}: {variable.Path}");
                        return;
                    }
                }

                ValidateOptimizationSetup();
            }));
        }

        // FUNCIONES PARA AGREGAR Y REMOVER FUNCIONES OBJETIVO Y RESTRICCIONES
        private void BtnAddObjective_Click(object sender, RoutedEventArgs e)
        {
            ObjectiveFunction objective = new ObjectiveFunction
            {
                Name = $"Obj{objectiveFunctions.Count + 1}",
                Expression = "Define Objective Function",
                Sense = "Minimize"
            };

            objectiveFunctions.Add(objective);
            UpdateProblemType();
            AddLog("INFO", $"Added objective function {objective.Name}.");
        }


        private void BtnRemoveObjective_Click(object sender, RoutedEventArgs e)
        {
            if (ObjectiveFunctionsGrid.SelectedItem is ObjectiveFunction selectedObjective)
            {
                objectiveFunctions.Remove(selectedObjective);
                UpdateProblemType();
                AddLog("INFO", $"Removed objective function {selectedObjective.Name}.");
                return;
            }

            AddLog("WARNING", "Select an objective function before removing it.");
        }

        // FUNCIONES DEL PANEL
        private void BtnAddObjectiveFunction_Panel_Click(object sender, RoutedEventArgs e)
        {
            BtnAddObjective_Click(sender, e);
        }


        private void BtnRemoveObjectiveFunction_Panel_Click(object sender, RoutedEventArgs e)
        {
            BtnRemoveObjective_Click(sender, e);
        }

        private void ObjectiveFunctionsGrid_CellEditEnding(object sender, DataGridCellEditEndingEventArgs e)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                UpdateProblemType();
            }));
        }


        // BOTONES DE CONSTRAINTS
        private void BtnAddConstraint_Click(object sender, RoutedEventArgs e)
        {
            optimizationConstraints.Add(new OptimizationConstraint
            {
                LeftSide = "",
                Operator = "<=",
                RightSide = "",
                Description = ""
            });

            ValidateOptimizationSetup();
            AddLog("INFO", "Added a new optimization constraint.");
        }

        private void BtnRemoveConstraint_Click(object sender, RoutedEventArgs e)
        {
            if (ConstraintsGrid.SelectedItem is OptimizationConstraint selectedConstraint)
            {
                optimizationConstraints.Remove(selectedConstraint);
                ValidateOptimizationSetup();
                AddLog("INFO", "Removed optimization constraint.");
                return;
            }

            AddLog("WARNING", "Select a constraint before removing it.");
        }

        private void UpdateProblemType()
        {
            if (objectiveFunctions.Count <= 1)
            {
                TxtProblemType.Text = "Single-objective";
            }
            else
            {
                TxtProblemType.Text = "Multi-objective";
            }
            UpdateAvailableAlgorithms();
        }

        private void BtnAddManualInput_Click(object sender, RoutedEventArgs e)
        {
            AspenVariable variable = new AspenVariable
            {
                Tag = $"x{inputParameters.Count + 1}",
                Path = "Add route",
                Value = "0",
                Restar = "0",
                Type = "Input",
                Editable = true,
                VariableKind = "Continuous",
                LowerBound = "0",
                UpperBound = "1"
            };

            inputParameters.Add(variable);
            UpdateInputTags();
            ValidateOptimizationSetup();
            AddLog("INFO", $"Added manual input {variable.Tag}.");
        }

        private void BtnAddManualOutput_Click(object sender, RoutedEventArgs e)
        {
            AspenVariable variable = new AspenVariable
            {
                Tag = $"y{outputObjectives.Count + 1}",
                Path = "",
                Value = "0",
                Type = "Output",
                Editable = false
            };

            outputObjectives.Add(variable);
            ValidateOptimizationSetup();
            AddLog("INFO", $"Added manual output {variable.Tag}.");
        }

        private void UpdateInputTags()
        {
            InputTags.Clear();

            foreach (AspenVariable variable in inputParameters)
            {
                if (!string.IsNullOrWhiteSpace(variable.Tag))
                {
                    InputTags.Add(variable.Tag);
                }
            }

            LeftVariableColumn.ItemsSource = null;
            LeftVariableColumn.ItemsSource = InputTags;
        }


        private bool InputVariablesAreContinuousOrEmpty()
        {
            return inputParameters.Count == 0 ||
                   inputParameters.All(v => string.Equals(v.VariableKind, "Continuous", StringComparison.OrdinalIgnoreCase));
        }

        private bool InputVariablesAreAllContinuous()
        {
            return inputParameters.Count > 0 &&
                   inputParameters.All(v => string.Equals(v.VariableKind, "Continuous", StringComparison.OrdinalIgnoreCase));
        }

        private bool AlgorithmRequiresContinuousVariables(string algorithm)
        {
            return algorithm == "Trust Region Bayesian Optimization" ||
                   algorithm == "Vanilla M.O. Bayesian Optimization";
        }

        private bool InputVariablesAreContinuousDiscreteOrEmpty()
        {
            return inputParameters.Count == 0 ||
                   inputParameters.All(v =>
                       string.Equals(v.VariableKind, "Continuous", StringComparison.OrdinalIgnoreCase) ||
                       string.Equals(v.VariableKind, "Discrete", StringComparison.OrdinalIgnoreCase)
                   );
        }

        // Información sobre algoritmos
        private void UpdateAvailableAlgorithms()
        {
            AlgorithmComboBox.Items.Clear();
            bool continuousAlgorithmsAvailable = InputVariablesAreContinuousOrEmpty();
            bool morphBoAvailable = InputVariablesAreContinuousDiscreteOrEmpty();
            if (objectiveFunctions.Count <= 1)
            {
                AlgorithmComboBox.Items.Add("GA");
                AlgorithmComboBox.Items.Add("DE");
                AlgorithmComboBox.Items.Add("PSO");
                AlgorithmComboBox.Items.Add("NelderMead");
                AlgorithmComboBox.Items.Add("PatternSearch");
                AlgorithmComboBox.Items.Add("BRKGA");
                AlgorithmComboBox.Items.Add("ES");
                AlgorithmComboBox.Items.Add("SRES");
                AlgorithmComboBox.Items.Add("ISRES");
                AlgorithmComboBox.Items.Add("CMAES");
                AlgorithmComboBox.Items.Add("G3PCX");
                AlgorithmComboBox.Items.Add("NRBO");
                AlgorithmComboBox.Items.Add("SBOA");
                AlgorithmComboBox.Items.Add("Vanilla Bayesian Optimization");
                AlgorithmComboBox.Items.Add("Discrete Bayesian Optimization");
                if (continuousAlgorithmsAvailable)
                    AlgorithmComboBox.Items.Add("Trust Region Bayesian Optimization");
                TxtProblemType.Text = "Single-objective";
                //if (morphBoAvailable)
                //    AlgorithmComboBox.Items.Add("MORPHBO");
            }
            else
            {
                AlgorithmComboBox.Items.Add("NSGA-II");
                AlgorithmComboBox.Items.Add("R-NSGA-II");
                AlgorithmComboBox.Items.Add("NSGA-III");
                AlgorithmComboBox.Items.Add("U-NSGA-III");
                AlgorithmComboBox.Items.Add("R-NSGA-III");
                AlgorithmComboBox.Items.Add("MOEA/D");
                AlgorithmComboBox.Items.Add("AGE-MOEA");
                AlgorithmComboBox.Items.Add("AGE-MOEA2");
                AlgorithmComboBox.Items.Add("RVEA");
                AlgorithmComboBox.Items.Add("SMS-EMOA");
                AlgorithmComboBox.Items.Add("MOPSO-CD");
                AlgorithmComboBox.Items.Add("CMOPSO");
                //if (morphBoAvailable)
                //    AlgorithmComboBox.Items.Add("MORPHBO");

                if (continuousAlgorithmsAvailable)
                    AlgorithmComboBox.Items.Add("Vanilla M.O. Bayesian Optimization");
                TxtProblemType.Text = "Multi-objective";
            }

            AlgorithmComboBox.SelectedIndex = -1;
            BtnRunOptimization.IsEnabled = false;
            //TxtOptimizationStatus.Text = "Select an algorithm.";
            SetActivityStatus("Select an optimization algorithm.");
        }


        private void ValidateOptimizationSetup()
        {
            bool hasInputs = inputParameters.Count > 0;
            bool hasOutputs = outputObjectives.Count > 0;
            bool hasObjectives = objectiveFunctions.Count > 0;
            bool hasAlgorithm = AlgorithmComboBox.SelectedItem != null;
            string selectedAlgorithm = AlgorithmComboBox.SelectedItem?.ToString() ?? "";
            bool validContinuousOnlyAlgorithm =
                !AlgorithmRequiresContinuousVariables(selectedAlgorithm) || InputVariablesAreAllContinuous();

            bool validInputBounds = inputParameters.All(v =>
            {
                string lbText = v.LowerBound.Replace(",", ".");
                string ubText = v.UpperBound.Replace(",", ".");

                bool lbOk = double.TryParse(
                    lbText,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out double lb
                );

                bool ubOk = double.TryParse(
                    ubText,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out double ub
                );

                if (!lbOk || !ubOk)
                    return false;

                if (ub <= lb)
                    return false;

                if (v.VariableKind == "Discrete")
                    return lb % 1 == 0 && ub % 1 == 0;

                return !string.IsNullOrWhiteSpace(v.Tag);
            });

            bool validOutputs = outputObjectives.All(v =>
                !string.IsNullOrWhiteSpace(v.Tag));

            bool validObjectives = objectiveFunctions.All(f =>
                !string.IsNullOrWhiteSpace(f.Name) &&
                !string.IsNullOrWhiteSpace(f.Expression) &&
                !string.IsNullOrWhiteSpace(f.Sense));

            bool validConstraints = optimizationConstraints.All(c =>
                string.IsNullOrWhiteSpace(c.LeftSide) ||
                (
                    !string.IsNullOrWhiteSpace(c.LeftSide) &&
                    !string.IsNullOrWhiteSpace(c.Operator) &&
                    !string.IsNullOrWhiteSpace(c.RightSide)
                ));

            bool validHyperparameters =
                hyperparameterTextBoxes.All(h => !string.IsNullOrWhiteSpace(h.Value.Text)) &&
                hyperparameterComboBoxes.All(h => h.Value.SelectedItem != null);

            bool ready =
                hasInputs &&
                hasOutputs &&
                hasObjectives &&
                hasAlgorithm &&
                validInputBounds &&
                validOutputs &&
                validObjectives &&
                validConstraints &&
                validHyperparameters &&
                validContinuousOnlyAlgorithm;

            BtnRunOptimization.IsEnabled = ready;

            if (ready)
            {
                //TxtOptimizationStatus.Text = "Ready to run.";
                SetActivityStatus("Ready to run.");
            }
            else if (hasAlgorithm && !validContinuousOnlyAlgorithm)
            {
                //TxtOptimizationStatus.Text = "Selected Bayesian trust-region/MOBO method only supports continuous variables.";
                SetActivityStatus("Selected algorithm requires continuous variables.");
            }
            else
            {
                //TxtOptimizationStatus.Text = "Incomplete setup.";
                SetActivityStatus("Incomplete optimization setup.");
            }
        }


        // Esta funcion solo es para cuando se da click a un metodo
        private void AlgorithmComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            LoadHyperparametersForSelectedAlgorithm();
            ValidateOptimizationSetup();

            string algorithm = AlgorithmComboBox.SelectedItem?.ToString() ?? "";
            if (!string.IsNullOrWhiteSpace(algorithm))
            {
                AddLog("INFO", $"Optimization algorithm selected: {algorithm}.");
            }
        }

        private void LoadHyperparametersForSelectedAlgorithm()
        {
            HyperparameterPanel.Children.Clear();
            hyperparameterTextBoxes.Clear();
            hyperparameterComboBoxes.Clear();

            string algorithm = AlgorithmComboBox.SelectedItem?.ToString() ?? "";
            string defaultRefPoint = string.Join(",", Enumerable.Repeat("0.5", Math.Max(2, objectiveFunctions.Count)));
            // Ajustar estooo , puede haber error
            if (algorithm == "PSO")  //(algorithm.Contains("PSO"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("w", "Inertia Weight", "0.9");
                AddNumericHyperparameter("c1", "Cognitive Coefficient", "2.0");
                AddNumericHyperparameter("c2", "Social Coefficient", "2.0");
                AddBooleanHyperparameter("adaptive", "Adaptive Parameters", true);
                AddTextHyperparameter("initial_velocity", "Initial Velocity", "random");
                AddNumericHyperparameter("max_velocity_rate", "Maximum Velocity Rate", "0.20");
                AddBooleanHyperparameter("pertube_best", "Perturb Best Particle", true);
            }
            else if (algorithm == "DE") //(algorithm.Contains("DE"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("n_offsprings", "Offsprings", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "100");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("sampling", "Sampling", "real_random");
            }
            else if (algorithm == "GA")  //(algorithm.Contains("GA"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("offsprings", "Offsprings", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "100");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("sampling", "Sampling", "real_random");
                AddTextHyperparameter("selection", "Selection", "tournament");
                AddTextHyperparameter("crossover", "Crossover", "SBX");
                AddTextHyperparameter("mutation", "Mutation", "PM");
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", true);
            }
            else if (algorithm == "NelderMead")  //(algorithm.Contains("NelderMead"))
            {

                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("simplex", "Simplex", "0.02");
                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "PatternSearch") //(algorithm.Contains("PatternSearch"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("delta", "Delta", "0.25");
                AddNumericHyperparameter("rho", "Rho", "0.5");
                AddNumericHyperparameter("step_size", "Step size", "1.0");
            }
            else if (algorithm == "BRKGA") //(algorithm.Contains("BRKGA"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("n_elites", "Elite Proportion", "50");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "50");
                AddNumericHyperparameter("n_mutants", "Mutant Proportion", "100");
                AddNumericHyperparameter("crossover_bias", "Crossover Bias", "0.7");
                AddTextHyperparameter("sampling", "Sampling", "real_random");
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", false);
            }
            else if (algorithm == "ES") //(algorithm.Contains("ES"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("pop_size", "Population Size", "50");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "50");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("rule", "Rule", "0.12");
                AddNumericHyperparameter("phi", "Phi", "1.0");
                AddNumericHyperparameter("gamma", "Gamma", "0.85");
                AddTextHyperparameter("sampling", "Sampling", "real_random");
            }
            else if (algorithm == "SRES") // (algorithm.Contains("SRES"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("pf", "PF", "0.45");
                AddNumericHyperparameter("seed", "Seed", "42");

            }
            else if (algorithm == "ISRES") //(algorithm.Contains("ISRES"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "50");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("rule", "Rule", "0.12");
                AddNumericHyperparameter("gamma", "Gamma", "0.85");
                AddNumericHyperparameter("alpha", "Alpha", "1.0");

            }
            else if (algorithm == "CMAES") //(algorithm.Contains("CMAES"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("n_init", "Initial samples", "20");
                AddNumericHyperparameter("sigma", "Sigma", "0.1");
                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "G3PCX") //(algorithm.Contains("G3PCX"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("pop_size", "Population size", "100");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "2");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("n_parents", "Rule", "3");
                AddNumericHyperparameter("family_size", "Family size", "2");
                AddTextHyperparameter("sampling", "Sampling", "real_random");
            }
            else if (algorithm == "NRBO") //(algorithm.Contains("NRBO"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("pop_size", "Population size", "50");
                AddNumericHyperparameter("deciding_factor", "Deciding factor", "2");
                AddNumericHyperparameter("max_iteration", "Max iterations", "3");
                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "SBOA") //(algorithm.Contains("SBOA"))
            {
                AddNumericHyperparameter("pop_size", "Population size", "30");
                AddNumericHyperparameter("max_iter", "Max iterations", "10");
                AddNumericHyperparameter("Cp1", "Exploration coefficient (Cp1)", "0.9");
                AddNumericHyperparameter("Cp2", "Exploitation coefficient (Cp2)", "0.55");
                AddNumericHyperparameter("seed", "Seed", "42");

            }
            else if (algorithm == "NSGA-II") //(algorithm.Contains("NSGA-II"))
            {
                AddNumericHyperparameter("generations", "Generations", "50");
                AddNumericHyperparameter("pop_size", "Population Size", "100");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "50");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("sampling", "Sampling", "real_random");
                AddTextHyperparameter("crossover", "Crossover", "SBX");
                AddTextHyperparameter("mutation", "Mutation", "PM");
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", true);
            }
            else if (algorithm == "Vanilla Bayesian Optimization") //\(algorithm.Contains("Vanilla Bayesian Optimization"))
            {
                AddNumericHyperparameter("n_init", "Initial samples", "5");
                AddNumericHyperparameter("n_iter", "BO iterations", "30");
                AddStackedHyperparameter("aqf", "Acquisition function");
                //AddNumericHyperparameter("n_candidates", "Candidate points", "2000");
                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "Trust Region Bayesian Optimization")
            {
                // CHEQUEAR SI TODOS LOS PARAMETROS QUE VOY A AGREGAR SON ESTOS. SON MUCHO
                AddNumericHyperparameter("n_init", "Initial samples", "10");
                AddNumericHyperparameter("n_iter", "TuRBO iterations", "30");
                AddStackedHyperparameter("aqf", "Acquisition function");
                //AddNumericHyperparameter("batch_size", "Batch size", "1");
                //AddNumericHyperparameter("n_candidates", "Candidate points", "2000");
                //AddNumericHyperparameter("num_restarts", "Acquisition restarts", "10");
                //AddNumericHyperparameter("raw_samples", "Raw samples", "512");
                AddNumericHyperparameter("length", "Initial trust-region length", "0.5");
                AddNumericHyperparameter("length_min", "Minimum trust-region length", "0.000000001");
                AddNumericHyperparameter("length_max", "Maximum trust-region length", "1.6");
                AddNumericHyperparameter("success_tolerance", "Success tolerance", "3");
                AddNumericHyperparameter("failure_tolerance", "Failure tolerance", "-1");
                AddNumericHyperparameter("shrink_factor", "Shrink factor", "0.7");
                AddNumericHyperparameter("expand_factor", "Expand factor", "2.0");
                //AddNumericHyperparameter("abs_tol", "Absolute improvement tolerance", "0.000001");
                //AddNumericHyperparameter("rel_tol", "Relative improvement tolerance", "0.001");
                //AddNumericHyperparameter("beta", "UCB beta", "0.5");
                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "MORPHBO")
            {
                AddNumericHyperparameter("n_init", "Initial samples", "12");
                AddNumericHyperparameter("n_iter", "MORPHBO iterations", "30");

                //AddStackedHyperparameter("aqf", "Acquisition function");
                if (objectiveFunctions.Count <= 1)
                {
                    AddStackedHyperparameter("aqf", "Acquisition function");
                }
                //AddNumericHyperparameter("batch_size", "Batch size", "1");
                //AddNumericHyperparameter("n_candidates", "Candidate points", "2000");
                //AddNumericHyperparameter("global_candidate_fraction", "Global candidate fraction", "0.15");

                //AddTextHyperparameter("kernel_mode", "Kernel mode (mixed/matern)", "mixed");

                AddNumericHyperparameter("length", "Initial continuous TR length", "0.5");
                AddNumericHyperparameter("length_min", "Minimum continuous TR length", "0.000001");
                AddNumericHyperparameter("length_max", "Maximum continuous TR length", "1.0");

                AddNumericHyperparameter("discrete_radius", "Initial discrete radius", "3");
                AddNumericHyperparameter("discrete_radius_min", "Minimum discrete radius", "1");
                AddNumericHyperparameter("discrete_radius_max", "Maximum discrete radius", "10");

                AddNumericHyperparameter("success_tolerance", "Success tolerance", "3");
                AddNumericHyperparameter("failure_tolerance", "Failure tolerance (-1 auto)", "-1");

                AddNumericHyperparameter("shrink_factor", "Shrink factor", "0.7");
                AddNumericHyperparameter("expand_factor", "Expand factor", "1.5");

                //AddNumericHyperparameter("abs_tol", "Absolute improvement tolerance", "0.000001");
                //AddNumericHyperparameter("rel_tol", "Relative improvement tolerance", "0.001");

                //AddNumericHyperparameter("beta", "UCB beta", "0.5");

                //AddTextHyperparameter("use_feasibility", "Use feasibility model (true/false)", "true");
                //AddNumericHyperparameter("feasibility_rho", "Feasibility penalty rho", "1.0");
                //AddNumericHyperparameter("p_feas_min", "Minimum feasibility probability", "0.001");
                AddNumericHyperparameter("min_feasibility_points", "Minimum feasibility points", "5");

                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "Discrete Bayesian Optimization") //(algorithm.Contains("Discrete Bayesian Optimization"))
            {
                AddNumericHyperparameter("n_init", "Initial samples", "5");
                AddNumericHyperparameter("n_iter", "BO iterations", "30");
                AddNumericHyperparameter("beta", "Initial beta", "2.0");
                AddNumericHyperparameter("beta_h", "Beta upper limit", "25.0");
                AddNumericHyperparameter("lengthscale_min", "Lengthscale min", "0.05");
                AddNumericHyperparameter("lengthscale_max", "Lengthscale max", "2.0");
                AddNumericHyperparameter("lengthscale_trials", "Lengthscale trials", "6");
                //AddNumericHyperparameter("n_candidates", "Candidate points", "2000");
                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "Vanilla M.O. Bayesian Optimization")
            {
                AddNumericHyperparameter("n_init", "Initial samples", "10");
                AddNumericHyperparameter("n_iter", "MOBO iterations", "30");
                AddNumericHyperparameter("batch_size", "Batch size", "1");
                AddTextHyperparameter("ref_point", "Reference point in Y=-F space", "auto");
                AddNumericHyperparameter("num_restarts", "Acquisition restarts", "10");
                AddNumericHyperparameter("raw_samples", "Raw samples", "512");
                AddNumericHyperparameter("seed", "Seed", "42");
            }
            else if (algorithm == "R-NSGA-II")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("ref_points", "Reference/Aspiration points", defaultRefPoint);
                AddNumericHyperparameter("epsilon", "Epsilon", "0.001");
                AddTextHyperparameter("normalization", "Normalization", "front");
                AddBooleanHyperparameter("extreme_points_as_reference_points", "Use extreme points as reference", false);

            }
            else if (algorithm == "NSGA-III")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size / Reference directions", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("ref_dir_method", "Reference direction method", "energy");
                AddNumericHyperparameter("n_ref_dirs", "Number of reference directions", "25");
                AddNumericHyperparameter("n_partitions", "Das-Dennis partitions", "12");

            }
            else if (algorithm == "U-NSGA-III")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("ref_dir_method", "Reference direction method", "energy");
                AddNumericHyperparameter("n_ref_dirs", "Number of reference directions", "25");
                AddNumericHyperparameter("n_partitions", "Das-Dennis partitions", "12");
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", true);

            }
            else if (algorithm == "R-NSGA-III")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("ref_points", "Reference/Aspiration points", defaultRefPoint);
                AddNumericHyperparameter("pop_per_ref_point", "Population per reference point", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "10");
                AddNumericHyperparameter("mu", "Mu", "0.05");

            }
            else if (algorithm == "MOEA/D")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("ref_dir_method", "Reference direction method", "energy");
                AddNumericHyperparameter("n_ref_dirs", "Number of reference directions", "25");
                AddNumericHyperparameter("n_partitions", "Das-Dennis partitions", "12");
                AddNumericHyperparameter("n_neighbors", "Neighbors", "15");
                AddNumericHyperparameter("prob_neighbor_mating", "Neighbor mating probability", "0.7");

            }
            else if (algorithm == "AGE-MOEA")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", true);

            }
            else if (algorithm == "AGE-MOEA2")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", true);

            }
            else if (algorithm == "RVEA")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "10");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddTextHyperparameter("ref_dir_method", "Reference direction method", "energy");
                AddNumericHyperparameter("n_ref_dirs", "Number of reference directions", "25");
                AddNumericHyperparameter("n_partitions", "Das-Dennis partitions", "12");
                AddNumericHyperparameter("alpha", "Alpha", "2.0");
                AddNumericHyperparameter("adapt_freq", "Adaptation frequency", "0.1");
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", true);

            }
            else if (algorithm == "SMS-EMOA")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("n_offsprings", "n_offsprings", "1");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddBooleanHyperparameter("normalize", "Normalize objectives", true);
                AddBooleanHyperparameter("eliminate_duplicates", "Eliminate duplicates", true);

            }
            else if (algorithm == "MOPSO-CD")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("w", "Inertia Weight", "0.6");
                AddNumericHyperparameter("c1", "Cognitive Coefficient", "2.0");
                AddNumericHyperparameter("c2", "Social Coefficient", "2.0");
                AddNumericHyperparameter("max_velocity_rate", "Maximum Velocity Rate", "0.50");
                AddNumericHyperparameter("archive_size", "Archive Size", "100");

            }
            else if (algorithm == "CMOPSO")
            {
                AddNumericHyperparameter("generations", "Generations", "10");
                AddNumericHyperparameter("pop_size", "Population Size", "25");
                AddNumericHyperparameter("seed", "Seed", "42");
                AddNumericHyperparameter("max_velocity_rate", "Maximum Velocity Rate", "0.20");
                AddNumericHyperparameter("elite_size", "Elite Size", "10");
                AddTextHyperparameter("initial_velocity", "Initial Velocity", "random");
                AddNumericHyperparameter("mutation_rate", "Mutation Rate", "0.50");

            }

        }

        private void AddNumericHyperparameter(string name, string label, string defaultValue)
        {
            StackPanel row = CreateHyperparameterRow(label);

            TextBox textBox = new TextBox
            {
                Text = defaultValue,
                Width = 210,
                Height = 24,
                HorizontalAlignment = HorizontalAlignment.Left
                //Margin = new Thickness(0, 2, 0, 2)
            };

            textBox.TextChanged += (s, e) => ValidateOptimizationSetup();


            row.Children.Add(textBox);

            hyperparameterTextBoxes[name] = textBox;
            HyperparameterPanel.Children.Add(row);
        }

        private void AddTextHyperparameter(string name, string label, string defaultValue)
        {
            StackPanel row = CreateHyperparameterRow(label);

            TextBox textBox = new TextBox
            {
                Text = defaultValue,
                Width = 210,
                Height = 24,
                HorizontalAlignment = HorizontalAlignment.Left
            };

            textBox.TextChanged += (s, e) => ValidateOptimizationSetup();

            row.Children.Add(textBox);

            hyperparameterTextBoxes[name] = textBox;
            HyperparameterPanel.Children.Add(row);
        }

        private void AddBooleanHyperparameter(string key, string label, bool defaultValue)
        {
            StackPanel row = CreateHyperparameterRow(label);

            ComboBox comboBox = new ComboBox
            {
                Width = 210,
                Height = 24,
                HorizontalAlignment = HorizontalAlignment.Left
            };

            comboBox.Items.Add("true");
            comboBox.Items.Add("false");
            comboBox.SelectedItem = defaultValue ? "true" : "false";

            comboBox.SelectionChanged += (s, e) => ValidateOptimizationSetup();

            row.Children.Add(comboBox);

            hyperparameterComboBoxes[key] = comboBox;
            HyperparameterPanel.Children.Add(row);
        }

        private void AddStackedHyperparameter(string key, string label)
        {
            StackPanel row = CreateHyperparameterRow(label);

            ComboBox comboBox = new ComboBox
            {
                Width = 210,
                Height = 24,
                HorizontalAlignment = HorizontalAlignment.Left
            };

            comboBox.Items.Add("Log-EI");
            comboBox.Items.Add("Upper Confidence Bound");
            comboBox.Items.Add("Thomson Sampling");

            comboBox.SelectionChanged += (s, e) =>
            {
                string selected = comboBox.SelectedItem?.ToString() ?? "";

                if (!string.IsNullOrWhiteSpace(selected))
                {
                    LoadAcquisitionHyperparameters(selected);
                }

                ValidateOptimizationSetup();
            };

            row.Children.Add(comboBox);

            hyperparameterComboBoxes[key] = comboBox;
            HyperparameterPanel.Children.Add(row);
        }

        private void RemoveHyperparameterIfExists(string key)
        {
            if (!hyperparameterTextBoxes.ContainsKey(key))
                return;

            TextBox textBox = hyperparameterTextBoxes[key];

            StackPanel? rowToRemove = null;

            foreach (UIElement element in HyperparameterPanel.Children)
            {
                if (element is StackPanel row && row.Children.Contains(textBox))
                {
                    rowToRemove = row;
                    break;
                }
            }

            if (rowToRemove != null)
            {
                HyperparameterPanel.Children.Remove(rowToRemove);
            }

            hyperparameterTextBoxes.Remove(key);
        }

        private void LoadAcquisitionHyperparameters(string acquisition)
        {
            // Remueve solo hiperparámetros específicos de adquisición
            RemoveHyperparameterIfExists("beta");
            //RemoveHyperparameterIfExists("num_samples");
            //RemoveHyperparameterIfExists("raw_samples");
            //RemoveHyperparameterIfExists("mc_samples");
            //RemoveHyperparameterIfExists("best_f");

            //if (acquisition == "Log-EI")
            //{
            //    //AddNumericHyperparameter("raw_samples", "Raw samples", "512");
            //    //AddNumericHyperparameter("best_f", "Best observed value", "auto");
            //}
            if (acquisition == "Upper Confidence Bound")
            {
                AddNumericHyperparameter("beta", "Beta", "2.0");
                //AddNumericHyperparameter("raw_samples", "Raw samples", "512");
            }
            //else if (acquisition == "Thomson Sampling")
            //{
            //    //AddNumericHyperparameter("num_samples", "Thomson samples", "128");
            //    //AddNumericHyperparameter("raw_samples", "Raw samples", "512");
            //}

            ValidateOptimizationSetup();
        }

        // CREA EL PANEL DE HYPERPARAMETERS CON EL LABEL Y EL CONTROL DE INPUT
        private StackPanel CreateHyperparameterRow(string label)
        {
            StackPanel panel = new StackPanel
            {
                Orientation = Orientation.Vertical,
                Margin = new Thickness(10),
                Width = 260
            };

            TextBlock textBlock = new TextBlock
            {
                Text = label,
                Margin = new Thickness(0, 0, 0, 5)
            };

            panel.Children.Add(textBlock);

            return panel;
        }

        // Funcion para extraer hyperparametros 
        private Dictionary<string, string> ExtractHyperparameters()
        {
            Dictionary<string, string> hyperparameters =
                new Dictionary<string, string>();

            foreach (var item in hyperparameterTextBoxes)
            {
                hyperparameters[item.Key] = item.Value.Text;
            }

            foreach (var item in hyperparameterComboBoxes)
            {
                hyperparameters[item.Key] =
                    item.Value.SelectedItem?.ToString() ?? "";
            }

            return hyperparameters;
        }

        private async void BtnRunOptimization_Click(object sender, RoutedEventArgs e)
        {
            ValidateOptimizationSetup();

            if (!BtnRunOptimization.IsEnabled)
            {
                AddLog("WARNING", "The optimization configuration is incomplete.");
                SetActivityStatus("Incomplete optimization setup.");
                return;
            }

            BtnRunOptimization.IsEnabled = false;
            BtnExportResults.IsEnabled = false;
            //TxtOptimizationStatus.Text = "Running optimization...";
            SetActivityStatus("Optimization running...");
            SetProgress(null, true);
            lastLoggedBatchProgress = -1;

            try
            {
                string algorithm = AlgorithmComboBox.SelectedItem?.ToString() ?? "";
                Dictionary<string, string> hyperparameters = ExtractHyperparameters();
                string backend = UsesCSharpAspenBackend(algorithm) ? "C# → Aspen / JSONL" : "Python → Aspen COM";

                AddLog("INFO", $"Starting {algorithm} optimization.");
                AddLog("INFO", $"Evaluation backend: {backend}.");
                AddLog("INFO", $"Inputs: {inputParameters.Count}; outputs: {outputObjectives.Count}; objectives: {objectiveFunctions.Count}; constraints: {optimizationConstraints.Count}.");

                var optimizationData = new
                {
                    aspen_file = selectedAspenFile,
                    problem_type = objectiveFunctions.Count <= 1 ? "single_objective" : "multi_objective",
                    algorithm = algorithm,
                    evaluation_backend = UsesCSharpAspenBackend(algorithm) ? "csharp_jsonl" : "python_com",
                    hyperparameters = hyperparameters,

                    inputs = inputParameters.Select(v => new
                    {
                        tag = v.Tag,
                        path = v.Path,
                        current_value = v.Value,
                        restart_values = string.IsNullOrWhiteSpace(v.Restar) ? v.Value : v.Restar,
                        variable_type = v.VariableKind,
                        lower_bound = double.Parse(v.LowerBound.Replace(",", "."), CultureInfo.InvariantCulture),
                        upper_bound = double.Parse(v.UpperBound.Replace(",", "."), CultureInfo.InvariantCulture)
                    }).ToList(),

                    outputs = outputObjectives.Select(v => new
                    {
                        tag = v.Tag,
                        path = v.Path,
                        current_value = v.Value
                    }).ToList(),

                    objectives = objectiveFunctions.Select(f => new
                    {
                        name = f.Name,
                        expression = f.Expression,
                        sense = f.Sense
                    }).ToList(),

                    constraints = optimizationConstraints
                        .Where(c => !string.IsNullOrWhiteSpace(c.LeftSide))
                        .Select(c => new
                        {
                            left_side = c.LeftSide,
                            op = c.Operator,
                            right_side = c.RightSide,
                            description = c.Description
                        }).ToList()
                };

                string dataDirectory = GetMplDataDirectory();
                string configPath = Path.Combine(dataDirectory, "optimization_config.json");
                string resultsPath = Path.Combine(dataDirectory, "optimization_results.json");

                if (File.Exists(resultsPath))
                {
                    File.Delete(resultsPath);
                }

                string json = JsonConvert.SerializeObject(optimizationData, Formatting.Indented);
                File.WriteAllText(configPath, json);
                AddLog("INFO", $"Optimization configuration written to {configPath}");

                string resultJson;

                if (UsesCSharpAspenBackend(algorithm))
                {
                    resultJson = await RunPythonOptimizerWithCSharpAspenAsync(
                        configPath,
                        resultsPath,
                        json);
                }
                else
                {
                    resultJson = await RunPythonOptimizerAsync(configPath, resultsPath);
                }

                lastResultsPath = resultsPath;
                lastResultsJson = resultJson;

                //TxtOptimizationStatus.Text = "Optimization finished.";
                SetActivityStatus("Optimization completed.");
                SetProgress(100);
                BtnExportResults.IsEnabled = true;
                AddLog("SUCCESS", $"Optimization finished successfully. Results: {resultsPath}");
            }
            catch (Exception ex)
            {
                //TxtOptimizationStatus.Text = "Optimization failed.";
                SetActivityStatus("Optimization failed.");
                SetProgress(0);
                AddLog("ERROR", $"Optimization failed: {ex.Message}");
            }
            finally
            {
                ValidateOptimizationSetup();
            }
        }


        private static bool UsesCSharpAspenBackend(string algorithm)
        {
            // Phase 1 final architecture: every algorithm currently exposed in the UI
            // delegates physical Aspen evaluations to the C# JSONL evaluation backend.
            string[] csharpAlgorithms =
            {
                // Single-objective
                "GA",
                "DE",
                "PSO",
                "NelderMead",
                "PatternSearch",
                "BRKGA",
                "ES",
                "SRES",
                "ISRES",
                "CMAES",
                "G3PCX",
                "NRBO",
                "SBOA",
                "Vanilla Bayesian Optimization",
                "Discrete Bayesian Optimization",
                "Trust Region Bayesian Optimization",

                // Multi-objective
                "NSGA-II",
                "R-NSGA-II",
                "NSGA-III",
                "U-NSGA-III",
                "R-NSGA-III",
                "MOEA/D",
                "AGE-MOEA",
                "AGE-MOEA2",
                "RVEA",
                "SMS-EMOA",
                "MOPSO-CD",
                "CMOPSO",
                "Vanilla M.O. Bayesian Optimization"
            };

            return csharpAlgorithms.Contains(algorithm, StringComparer.OrdinalIgnoreCase);
        }


        private async Task<string> RunPythonOptimizerWithCSharpAspenAsync(
            string configPath,
            string resultsPath,
            string configJson)
        {
            string pythonExe = "python";
            string scriptPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "optimizer_runner.py"
            );

            OptimizationBridgeConfig bridgeConfig =
                JsonConvert.DeserializeObject<OptimizationBridgeConfig>(configJson)
                ?? throw new InvalidOperationException("Could not deserialize the optimization bridge configuration.");

            await using AspenPlusProvider provider = new AspenPlusProvider();

            //TxtOptimizationStatus.Text = "Opening Aspen Plus from C#...";
            SetActivityStatus("Opening Aspen Plus from C#...");
            SetProgress(null, true);
            AddLog("INFO", "Opening Aspen Plus through the C# evaluation provider...");

            await provider.OpenAsync(selectedAspenFile);
            AddLog("SUCCESS", "Aspen Plus opened by the C# provider.");

            AspenEvaluationEngine engine = new AspenEvaluationEngine(provider, bridgeConfig);

            Progress<EvaluationProgress> progress = new Progress<EvaluationProgress>(p =>
            {
                string progressText =
                    $"Evaluating Aspen batch: {p.Completed}/{p.Total} " +
                    $"(OK: {p.Successful}, Failed: {p.Failed})";

                //TxtOptimizationStatus.Text = progressText;
                SetActivityStatus(progressText);

                if (p.Total > 1)
                {
                    // Population-based algorithms send real batches. For those, show
                    // batch progress and keep only coarse progress entries in the log.
                    double percent = 100.0 * p.Completed / p.Total;
                    SetProgress(percent);

                    int rounded = (int)Math.Round(percent);
                    if (p.Completed <= 1)
                    {
                        lastLoggedBatchProgress = -1;
                    }

                    if (p.Completed == p.Total || rounded >= lastLoggedBatchProgress + 10)
                    {
                        AddLog("PROGRESS", progressText);
                        lastLoggedBatchProgress = rounded;
                    }
                }
                else if (p.Total <= 0)
                {
                    SetProgress(null, true);
                }
                // For one-point BO/SBOA requests, do not force the progress bar to 100%
                // and do not add a permanent "1/1" line. The optimizer status message
                // that follows carries the meaningful overall optimization progress.
            });

            Progress<OptimizerStatusMessage> optimizerStatus =
                new Progress<OptimizerStatusMessage>(HandleOptimizerStatus);

            string result = await PythonJsonlOptimizerBridge.RunAsync(
                pythonExe,
                scriptPath,
                configPath,
                resultsPath,
                engine,
                progress,
                statusProgress: optimizerStatus);

            AddLog("INFO", "C#/Python optimization bridge completed.");
            return result;
        }


        private async Task<string> RunPythonOptimizerAsync(string configPath, string resultsPath)
        {
            string pythonExe = "python";

            string scriptPath = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "optimizer_runner.py"
            );

            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" \"{configPath}\" \"{resultsPath}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using Process process = new Process
            {
                StartInfo = start,
                EnableRaisingEvents = true
            };

            StringBuilder output = new StringBuilder();
            StringBuilder error = new StringBuilder();

            SetActivityStatus("Python optimizer running...");
            SetProgress(null, true);
            AddLog("INFO", $"Starting Python optimizer: {Path.GetFileName(scriptPath)}");

            process.Start();
            AddLog("INFO", $"Python process started (PID {process.Id}).");

            Task outputTask = PumpProcessStreamAsync(process.StandardOutput, output, "PYTHON");
            Task errorTask = PumpProcessStreamAsync(process.StandardError, error, "WARNING");

            await process.WaitForExitAsync();
            await Task.WhenAll(outputTask, errorTask);

            AddLog("INFO", $"Python process exited with code {process.ExitCode}.");

            if (process.ExitCode != 0)
            {
                throw new Exception(
                    $"Python finished with exit code {process.ExitCode}. Review the activity log for stdout/stderr details."
                );
            }

            if (!File.Exists(resultsPath))
            {
                throw new Exception(
                    "Python finished successfully, but the results file was not generated. Review the activity log for details."
                );
            }

            return File.ReadAllText(resultsPath);
        }


        private string ConvertJsonToCsv(string json)
        {
            JToken root = JToken.Parse(json);

            List<Dictionary<string, string>> rows = ExtractRowsFromJson(root);

            if (rows.Count == 0)
                throw new Exception("El archivo JSON no contiene datos exportables.");

            List<string> headers = rows
                .SelectMany(r => r.Keys)
                .Distinct()
                .ToList();

            StringBuilder csv = new StringBuilder();

            csv.AppendLine(string.Join(",", headers.Select(EscapeCsv)));

            foreach (Dictionary<string, string> row in rows)
            {
                List<string> values = headers
                    .Select(header => row.ContainsKey(header) ? row[header] : "")
                    .Select(EscapeCsv)
                    .ToList();

                csv.AppendLine(string.Join(",", values));
            }

            return csv.ToString();
        }


        private List<Dictionary<string, string>> ExtractRowsFromJson(JToken root)
        {
            List<Dictionary<string, string>> rows = new List<Dictionary<string, string>>();

            if (root is JArray rootArray)
            {
                foreach (JToken item in rootArray)
                {
                    rows.Add(FlattenJson(item));
                }

                return rows;
            }

            if (root is JObject rootObject)
            {
                string[] possibleArrayNames =
                {
            "history_records",
            "history",
            "evaluations",
            "all_evaluations",
            "records",
            "results",
            "data"
        };

                foreach (string arrayName in possibleArrayNames)
                {
                    JToken? candidate = rootObject[arrayName];

                    if (candidate is JArray array && array.Count > 0)
                    {
                        foreach (JToken item in array)
                        {
                            rows.Add(FlattenJson(item));
                        }

                        return rows;
                    }
                }

                foreach (JProperty property in rootObject.Properties())
                {
                    if (property.Value is JArray array && array.Count > 0)
                    {
                        foreach (JToken item in array)
                        {
                            rows.Add(FlattenJson(item));
                        }

                        return rows;
                    }
                }

                rows.Add(FlattenJson(rootObject));
            }

            return rows;
        }

        private Dictionary<string, string> FlattenJson(JToken token)
        {
            Dictionary<string, string> result = new Dictionary<string, string>();

            void Flatten(JToken currentToken, string prefix)
            {
                if (currentToken is JObject obj)
                {
                    foreach (JProperty property in obj.Properties())
                    {
                        string newPrefix = string.IsNullOrWhiteSpace(prefix)
                            ? property.Name
                            : prefix + "." + property.Name;

                        Flatten(property.Value, newPrefix);
                    }
                }
                else if (currentToken is JArray array)
                {
                    for (int i = 0; i < array.Count; i++)
                    {
                        string newPrefix = $"{prefix}_{i + 1}";
                        Flatten(array[i], newPrefix);
                    }
                }
                else
                {
                    result[prefix] = currentToken?.ToString() ?? "";
                }
            }

            Flatten(token, "");

            return result;
        }

        private string EscapeCsv(string value)
        {
            if (value == null)
                return "";

            bool mustQuote =
                value.Contains(",") ||
                value.Contains("\"") ||
                value.Contains("\n") ||
                value.Contains("\r");

            if (mustQuote)
            {
                value = value.Replace("\"", "\"\"");
                return "\"" + value + "\"";
            }

            return value;
        }


        private void BtnExportResults_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(selectedAspenFile))
                {
                    AddLog("WARNING", "Select an Aspen simulation before exporting results.");
                    return;
                }

                if (!File.Exists(selectedAspenFile))
                {
                    AddLog("ERROR", "The selected Aspen file was not found.");
                    return;
                }

                if (string.IsNullOrWhiteSpace(lastResultsPath) || !File.Exists(lastResultsPath))
                {
                    AddLog("WARNING", "There are no optimization results to export.");
                    return;
                }

                string userFileName = TxtFilenamePath.Text.Trim();

                if (string.IsNullOrWhiteSpace(userFileName))
                {
                    AddLog("WARNING", "Enter a name for the results file before exporting.");
                    return;
                }

                userFileName = Path.GetFileNameWithoutExtension(userFileName);

                foreach (char invalidChar in Path.GetInvalidFileNameChars())
                {
                    userFileName = userFileName.Replace(invalidChar, '_');
                }

                string aspenFolder = Path.GetDirectoryName(selectedAspenFile) ?? "";

                if (string.IsNullOrWhiteSpace(aspenFolder))
                {
                    AddLog("ERROR", "The Aspen simulation folder could not be resolved.");
                    return;
                }

                string exportPath = Path.Combine(aspenFolder, userFileName + ".csv");

                if (File.Exists(exportPath))
                {
                    MessageBoxResult overwrite = MessageBox.Show(
                        "A file with that name already exists.\n\nDo you want to replace it?",
                        "Confirm replacement",
                        MessageBoxButton.YesNo,
                        MessageBoxImage.Warning
                    );

                    if (overwrite != MessageBoxResult.Yes)
                    {
                        AddLog("INFO", "Results export was canceled by the user.");
                        return;
                    }
                }

                string json = File.ReadAllText(lastResultsPath);
                string csv = ConvertJsonToCsv(json);

                File.WriteAllText(
                    exportPath,
                    csv,
                    new UTF8Encoding(encoderShouldEmitUTF8Identifier: true)
                );

                AddLog("SUCCESS", $"Results exported successfully: {exportPath}");
                SetActivityStatus("Results exported.");
            }
            catch (Exception ex)
            {
                AddLog("ERROR", $"Error exporting results: {ex.Message}");
                SetActivityStatus("Export failed.");
            }
        }
    }
}