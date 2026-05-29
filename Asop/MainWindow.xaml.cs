using Asop.Aspen;
using Asop.Models;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;


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

         
        public MainWindow()
        {
            // ESTE ES EL CONSTRUCTOR DE LA CLASE, SE EJECUTA CUANDO SE CREA UNA INSTANCIA DE MainWindow
            InitializeComponent();
            DataContext = this;
            InputParametersGrid.ItemsSource = inputParameters;
            OutputObjectivesGrid.ItemsSource = outputObjectives;
            ObjectiveFunctionsGrid.ItemsSource = objectiveFunctions;
            ConstraintsGrid.ItemsSource = optimizationConstraints;
            LeftVariableColumn.ItemsSource = InputTags;
        }

        private void BtnFind_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();

            dialog.Title = "Selecciona un archivo Aspen";
            dialog.Filter = "Aspen Backup Files (*.bkp)|*.bkp";

            if (dialog.ShowDialog() == true)
            {
                selectedAspenFile = dialog.FileName;
                TxtFilePath.Text = selectedAspenFile;

                MessageBox.Show("Archivo seleccionado correctamente.");
            }
        }



        private async void BtnRead_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(selectedAspenFile))
            {
                MessageBox.Show("Primero selecciona un archivo Aspen.");
                return;
            }

            BtnRead.IsEnabled = false;

            bool success = connector.OpenSimulation(selectedAspenFile);

            if (!success)
            {
                MessageBox.Show("No se pudo abrir Aspen.");
                BtnRead.IsEnabled = true;
                return;
            }

            await Task.Delay(3000); // deja que Aspen termine de cargar

            List<AspenVariable> variables = await Task.Run(() =>
            {
                return connector.ReadTree();
            });

            allVariables = variables;
            BuildTreeView(allVariables);

            BtnRead.IsEnabled = true;

        }

        private void BtnClose_Click(object sender, RoutedEventArgs e)
        {
            connector.CloseSimulation();

            AspenTreeView.Items.Clear();
            inputParameters.Clear();
            outputObjectives.Clear();
            allVariables.Clear();
            TxtFilePath.Clear();
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
                MessageBox.Show("Selecciona una variable numérica del árbol.");
                return;
            }

            if (selectedVariable.Type == "Input")
            {
                bool alreadyExists = inputParameters.Any(v => v.Path == selectedVariable.Path);

                if (alreadyExists)
                {
                    MessageBox.Show("Esta variable ya fue agregada como parámetro de entrada.");
                    return;
                }

                double currentValue = 0.0;

                double.TryParse(
                    selectedVariable.Value,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out currentValue
                );

                AspenVariable newInput = new AspenVariable
                {
                    Tag = $"x{inputParameters.Count + 1}",
                    Path = selectedVariable.Path,
                    Value = selectedVariable.Value,
                    Type = selectedVariable.Type,
                    Editable = selectedVariable.Editable,
                    VariableKind = "Continuous",
                    LowerBound = currentValue * 0.5,
                    UpperBound = currentValue * 1.5
                    //Path = selectedVariable.Path,
                    //Value = selectedVariable.Value,
                    //Type = selectedVariable.Type,
                    //Editable = selectedVariable.Editable,
                    //VariableKind = "Continuous",
                    //LowerBound = currentValue * 0.5,
                    //UpperBound = currentValue * 1.5
                };

                inputParameters.Add(newInput);

                // Hacer update de los tags para que estén sincronizados con el orden en la grilla
                UpdateInputTags();


                MessageBox.Show("Variable agregada como parámetro de entrada.");
                return;
            }

            if (selectedVariable.Type == "Output")
            {
                bool alreadyExists = outputObjectives.Any(v => v.Path == selectedVariable.Path);

                if (alreadyExists)
                {
                    MessageBox.Show("Esta variable ya fue agregada como función objetivo.");
                    return;
                }

                AspenVariable newOutput = new AspenVariable
                {
                    //Path = selectedVariable.Path,
                    //Value = selectedVariable.Value,
                    //Type = selectedVariable.Type,
                    //Editable = selectedVariable.Editable,
                    //VariableKind = selectedVariable.VariableKind,
                    //LowerBound = selectedVariable.LowerBound,
                    //UpperBound = selectedVariable.UpperBound
                    Tag = $"y{outputObjectives.Count + 1}",
                    Path = selectedVariable.Path,
                    Value = selectedVariable.Value,
                    Type = selectedVariable.Type,
                    Editable = selectedVariable.Editable

                };

                outputObjectives.Add(newOutput);

                MessageBox.Show("Variable agregada como función objetivo.");
                return;
            }

            MessageBox.Show("La variable seleccionada no es Input ni Output.");
        }


        private void BtnRemoveSelected_Click(object sender, RoutedEventArgs e)
        {
            if (InputParametersGrid.SelectedItem is AspenVariable selectedInput)
            {
                AspenVariable? itemToRemove = inputParameters
                    .FirstOrDefault(v => v.Path == selectedInput.Path);

                if (itemToRemove != null)
                {
                    inputParameters.Remove(itemToRemove);
                    UpdateInputTags();
                    InputParametersGrid.Items.Refresh();
                }

                return;
            }

            if (OutputObjectivesGrid.SelectedItem is AspenVariable selectedOutput)
            {
                AspenVariable? itemToRemove = outputObjectives
                    .FirstOrDefault(v => v.Path == selectedOutput.Path);

                if (itemToRemove != null)
                {
                    outputObjectives.Remove(itemToRemove);
                    OutputObjectivesGrid.Items.Refresh();
                }

                return;
            }

            MessageBox.Show("Selecciona una variable de Input Parameters o Output Objectives para remover.");
        }


        private void InputParametersGrid_CellEditEnding(object sender, DataGridCellEditEndingEventArgs e)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                foreach (AspenVariable variable in inputParameters)
                {
                    if (variable.UpperBound <= variable.LowerBound)
                    {
                        MessageBox.Show(
                            $"Upper Bound debe ser mayor que Lower Bound para:\n{variable.Path}");
                        return;
                    }

                    if (variable.VariableKind == "Discrete")
                    {
                        if (variable.LowerBound % 1 != 0 || variable.UpperBound % 1 != 0)
                        {
                            MessageBox.Show(
                                $"Para variables discretas, Lower Bound y Upper Bound deben ser enteros:\n{variable.Path}");
                            return;
                        }
                    }
                }
            }));
        }


        private void BtnAddObjective_Click(object sender, RoutedEventArgs e)
        {
            objectiveFunctions.Add(new ObjectiveFunction
            {
                Name = $"Obj{objectiveFunctions.Count + 1}",
                Expression = "",
                Sense = "Minimize"
            });

            UpdateProblemType();
        }

        private void BtnRemoveObjective_Click(object sender, RoutedEventArgs e)
        {
            if (ObjectiveFunctionsGrid.SelectedItem is ObjectiveFunction selectedObjective)
            {
                objectiveFunctions.Remove(selectedObjective);
                UpdateProblemType();
                return;
            }

            MessageBox.Show("Selecciona una función objetivo para remover.");
        }

        private void ObjectiveFunctionsGrid_CellEditEnding(object sender, DataGridCellEditEndingEventArgs e)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                UpdateProblemType();
            }));
        }

        private void BtnAddConstraint_Click(object sender, RoutedEventArgs e)
        {
            optimizationConstraints.Add(new OptimizationConstraint
            {
                LeftSide = "",
                Operator = "<=",
                RightSide = "",
                Description = ""
            });
        }

        private void BtnRemoveConstraint_Click(object sender, RoutedEventArgs e)
        {
            if (ConstraintsGrid.SelectedItem is OptimizationConstraint selectedConstraint)
            {
                optimizationConstraints.Remove(selectedConstraint);
                return;
            }

            MessageBox.Show("Selecciona una restricción para remover.");
        }

        //private void ConstraintsGrid_CellEditEnding(object sender, DataGridCellEditEndingEventArgs e)
        //{
        //    Dispatcher.BeginInvoke(new Action(() =>
        //    {
        //        foreach (OptimizationConstraint constraint in optimizationConstraints)
        //        {
        //            // Ignorar filas totalmente vacías o recién creadas
        //            bool emptyRow =
        //                string.IsNullOrWhiteSpace(constraint.LeftSide) &&
        //                string.IsNullOrWhiteSpace(constraint.RightSide);

        //            if (emptyRow)
        //                continue;

        //            if (string.IsNullOrWhiteSpace(constraint.LeftSide))
        //            {
        //                MessageBox.Show("Selecciona la variable izquierda de la restricción.");
        //                return;
        //            }

        //            if (string.IsNullOrWhiteSpace(constraint.Operator))
        //            {
        //                MessageBox.Show("Selecciona el operador de la restricción.");
        //                return;
        //            }

        //            if (string.IsNullOrWhiteSpace(constraint.RightSide))
        //            {
        //                MessageBox.Show("Define el lado derecho de la restricción, por ejemplo x2 o 24.");
        //                return;
        //            }

        //            bool rightIsInputTag = InputTags.Contains(constraint.RightSide);

        //            bool rightIsNumber = double.TryParse(
        //                constraint.RightSide,
        //                System.Globalization.NumberStyles.Float,
        //                System.Globalization.CultureInfo.InvariantCulture,
        //                out _
        //            );

        //            if (!rightIsInputTag && !rightIsNumber)
        //            {
        //                MessageBox.Show(
        //                    "El lado derecho debe ser una variable input existente, como x2, o un número, como 24.");
        //                return;
        //            }
        //        }
        //    }));
        //}

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
        }

        private void BtnAddManualInput_Click(object sender, RoutedEventArgs e)
        {
            inputParameters.Add(new AspenVariable
            {
                Tag = $"x{inputParameters.Count + 1}",
                Path = "",
                Value = "0",
                Type = "Input",
                Editable = true,
                VariableKind = "Continuous",
                LowerBound = 0,
                UpperBound = 1
            });
            UpdateInputTags();
        }

        private void BtnAddManualOutput_Click(object sender, RoutedEventArgs e)
        {
            outputObjectives.Add(new AspenVariable
            {
                Tag = $"y{outputObjectives.Count + 1}",
                Path = "",
                Value = "0",
                Type = "Output",
                Editable = false
            });
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

    }
}