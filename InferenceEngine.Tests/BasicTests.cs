using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;
using InferenceEngine.Core;

namespace InferenceEngine.Tests
{
    public class BasicTests
    {
        [Fact]
        public void InferenceEngineOptions_DefaultValues_ShouldBeSetCorrectly()
        {
            var options = new InferenceEngineOptions();
            
            Assert.Equal(string.Empty, options.ModelPath);
            Assert.True(options.UseGpuAcceleration);
            Assert.Equal(0, options.DeviceId);
            Assert.True(options.WarmupOnLoad);
        }

        [Fact]
        public void InferenceEngineOptions_Properties_ShouldBeSettable()
        {
            var options = new InferenceEngineOptions
            {
                ModelPath = "test.onnx",
                UseGpuAcceleration = false,
                DeviceId = 1,
                WarmupOnLoad = false
            };
            
            Assert.Equal("test.onnx", options.ModelPath);
            Assert.False(options.UseGpuAcceleration);
            Assert.Equal(1, options.DeviceId);
            Assert.False(options.WarmupOnLoad);
        }

        [Fact]
        public void BaseInferenceEngine_Constructor_WithNullOptions_ShouldThrowArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new TestEngine(null!, NullLogger.Instance));
        }

        [Fact]
        public void BaseInferenceEngine_Constructor_WithNullLogger_ShouldUseNullLogger()
        {
            var options = new InferenceEngineOptions { ModelPath = "test.onnx" };
            var engine = new TestEngine(options, null);
            Assert.NotNull(engine);
        }

        private class TestEngine : BaseInferenceEngine<string, string>
        {
            public TestEngine(InferenceEngineOptions options, Microsoft.Extensions.Logging.ILogger? logger) 
                : base(options, logger) { }

            protected override (System.Collections.Generic.List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>, object?) PreProcessWithContext(string input)
            {
                return (new(), null);
            }

            protected override string PostProcessWithContext(Microsoft.ML.OnnxRuntime.IDisposableReadOnlyCollection<Microsoft.ML.OnnxRuntime.DisposableNamedOnnxValue> output, object? context)
            {
                return "test";
            }
        }
    }
}