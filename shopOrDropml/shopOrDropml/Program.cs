// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using shopOrDropml;
using shopOrDropml.Data;
using System.Diagnostics;

string dir = Directory.GetCurrentDirectory();
Debug.WriteLine(dir);
string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

MLContext mlContext = new MLContext(seed: 0);
var model = Train(mlContext, _trainDataPath);
Evaluate(mlContext, model);
TestSinglePrediction(mlContext, model);

ITransformer Train(MLContext mlContext, string dataPath)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<PurchaseData>(dataPath, hasHeader: true, separatorChar: ',');
    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Satisfaction")
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DayEncoded", inputColumnName: "DayofWeek"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CategoryEncoded", inputColumnName: "Category"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "OnlineEncoded", inputColumnName: "Online"))
        .Append(mlContext.Transforms.Concatenate("Features", "DayEncoded", "CategoryEncoded", "ItemCost", "OnlineEncoded"))
        .Append(mlContext.Regression.Trainers.FastTree());
    var model = pipeline.Fit(dataView);
    return model;
}

void Evaluate(MLContext mlContext, ITransformer model)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<PurchaseData>(_testDataPath, hasHeader: true, separatorChar: ',');
    var predictions = model.Transform(dataView);
    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
}

void TestSinglePrediction(MLContext mlContext, ITransformer model)
{
    var predictionFunction = mlContext.Model.CreatePredictionEngine<PurchaseData, SatisfactionPrediction>(model);
    var purchaseSample = new PurchaseData()
    {
        DayofWeek = "sun",
        Category= "Apparel",
        ItemCost = 25,
        Satisfaction = 7,// To predict. Actual/Observed = 15.5
        Online = false
    };
    var prediction = predictionFunction.Predict(purchaseSample);
    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted satisfaction: {prediction.Satisfaction:0.####}");
    Console.WriteLine($"**********************************************************************");
}