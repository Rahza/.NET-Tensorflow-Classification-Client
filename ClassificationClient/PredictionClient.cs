using Google.Protobuf;
using Google.Protobuf.Collections;
using Grpc.Core;
using System;
using Tensorflow;
using Tensorflow.Serving;
using static Tensorflow.TensorShapeProto.Types;

namespace TF_NN_ClassificationClient
{
    public class PredictionClient
    {
        private const int TIMEOUT = 2000;

        private Channel _channel = null;
        private PredictionService.PredictionServiceClient _client = null;

        private string _host = null;

        private string _modelSpec = null;
        private string _signatureName = null;
        private string _inputsKey = null;

        public PredictionClient(string host, string modelSpec = "default", string signatureName = "classify_image", string inputsKey = "inputs")
        {
            _host = host;
            _modelSpec = modelSpec;
            _signatureName = signatureName;
            _inputsKey = inputsKey;
        }

        public void Open()
        {
            _channel = new Channel(_host, ChannelCredentials.Insecure);
            _client = new PredictionService.PredictionServiceClient(_channel);
        }

        public void Close()
        {
            _channel.ShutdownAsync().Wait();
        }

        public PredictionResult Predict(byte[] byteArray)
        {
            TensorProto tensorProto = this.GetTensorProto(byteArray);
            PredictRequest request = this.GetPredictRequest(tensorProto);

            DateTime deadline = DateTime.UtcNow.AddSeconds(TIMEOUT);

            PredictResponse response = _client.Predict(request, new CallOptions(deadline: deadline));

            return this.ParseResponse(response);
        }

        private PredictionResult ParseResponse(PredictResponse response)
        {
            var resultClasses = response.Outputs["classes"].Int64Val;
            long[] classes = new long[Math.Max(resultClasses.Count, 8)];
            response.Outputs["classes"].Int64Val.CopyTo(classes, 0);

            var resultScores = response.Outputs["scores"].FloatVal;
            float[] scores = new float[Math.Max(resultScores.Count, 8)];
            response.Outputs["scores"].FloatVal.CopyTo(scores, 0);

            return new PredictionResult() { PredictedClass = (int)classes[0], Scores = scores };
        }

        private PredictRequest GetPredictRequest(TensorProto tensorProto)
        {
            PredictRequest request = new PredictRequest();
            request.ModelSpec = new ModelSpec() { Name = _modelSpec, SignatureName = _signatureName };
            request.Inputs.Add(_inputsKey, tensorProto);

            return request;
        }

        private TensorProto GetTensorProto(byte[] byteArray)
        {
            ByteString imageData = ByteString.CopyFrom(byteArray);

            TensorProto tensorProto = new TensorProto();

            Dim dimBatch = new Dim() { Name = "batch", Size = 1 };
            Dim dimData = new Dim() { Name = "data", Size = 1 };
            RepeatedField<Dim> repeatedField = new RepeatedField<Dim>();
            repeatedField.Add(dimBatch);
            repeatedField.Add(dimData);

            TensorShapeProto tensorShape = new TensorShapeProto();
            tensorShape.Dim = repeatedField;

            tensorProto.TensorShape = tensorShape;

            tensorProto.Dtype = DataType.DtString;
            tensorProto.StringVal.Add(imageData);

            return tensorProto;
        }
    }
}
