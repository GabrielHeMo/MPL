using System;
using System.Collections.Generic;
using System.Text;

//namespace Asop.Models
//{
//    public class AspenVariable
//    {
//        public string Path { get; set; }

//        public string Value { get; set; }

//        public string Type { get; set; }

//        public bool Editable { get; set; }
//    }
//}

namespace Asop.Models
{
    public class AspenVariable
    {
        public string Tag { get; set; } = "";      // x1, x2, y1, y2

        public string Path { get; set; } = "";
        public string Value { get; set; } = "";
        public string Type { get; set; } = "";

        public string OutputFile { get; set; } = "";  // Archivo de salida para variables de salida
        public bool Editable { get; set; }

        // Nuevos campos para optimización
        public string VariableKind { get; set; } = "Continuous";
        public string LowerBound { get; set; } = "";
        public string UpperBound { get; set; } = "";
        public string Restar { get; set; } = "";
    }

    public class OptimizationConstraint
    {
        //public string VariableAlias { get; set; } = "";
        //public double Value { get; set; }
        //public string ConstraintType { get; set; } = "Input";
        //public string Description { get; set; } = "";
        //public string LeftExpression { get; set; } = "x1";
        //public string Operator { get; set; } = "<=";
        //public string RightExpression { get; set; } = "0";
        public string LeftSide { get; set; } = "";   // x1
        public string Operator { get; set; } = "<";      // <, <=, >, >=, ==
        public string RightSide { get; set; } = "";      // x2 o número
        public string Description { get; set; } = "";

    }

    public class ObjectiveFunction
    {
        public string Name { get; set; } = "Obj1";
        public string Expression { get; set; } = "";
        public string Sense { get; set; } = "Minimize";
    }

}