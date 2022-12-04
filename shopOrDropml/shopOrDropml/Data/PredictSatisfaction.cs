using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace shopOrDropml.Data
{
    public class PurchaseData
    {
        [LoadColumn(0)]
        public string? DayofWeek;

        [LoadColumn(1)]
        public string? Category;

        [LoadColumn(2)]
        public float ItemCost;

        [LoadColumn(3)]
        public float Satisfaction;

        [LoadColumn(4)]
        public bool Online;
    }

    public class SatisfactionPrediction
    {
        [ColumnName("Score")]
        public float Satisfaction;
    }
}
