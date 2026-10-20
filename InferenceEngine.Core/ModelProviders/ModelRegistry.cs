using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceEngine.Core.ModelProviders
{
    /// <summary>
    /// Registry of known models and their download URLs from popular model hubs.
    /// Allows engines to specify only a model name for commonly known models.
    /// </summary>
    public class ModelRegistry
    {
        private readonly Dictionary<string, ModelRegistryEntry> _entries;
        private readonly ILogger _logger;
        private static readonly string DefaultRegistryPath = Path.Combine(
            AppContext.BaseDirectory,
            "models.json");

        /// <summary>
        /// Creates a model registry from a JSON file.
        /// </summary>
        /// <param name="registryPath">Path to the JSON registry file. Uses default if null.</param>
        /// <param name="logger">Optional logger</param>
        public ModelRegistry(string? registryPath = null, ILogger? logger = null)
        {
            _logger = logger ?? NullLogger.Instance;
            var path = registryPath ?? DefaultRegistryPath;
            _entries = LoadRegistry(path);
        }

        /// <summary>
        /// Gets download URLs for a model by name.
        /// </summary>
        /// <param name="modelName">Name of the model file (e.g., "yolo11n.onnx")</param>
        /// <returns>Array of URLs to download the model from, or null if not found</returns>
        public string[]? GetModelUrls(string modelName)
        {
            if (_entries.TryGetValue(modelName, out var entry))
            {
                _logger.LogDebug("Found model '{ModelName}' in registry with {UrlCount} source(s)", modelName, entry.Urls.Length);
                return entry.Urls;
            }

            _logger.LogDebug("Model '{ModelName}' not found in registry", modelName);
            return null;
        }

        /// <summary>
        /// Gets the full registry entry for a model by name.
        /// </summary>
        /// <param name="modelName">Name of the model file (e.g., "yolo11n.onnx")</param>
        /// <returns>The registry entry, or null if not found</returns>
        public ModelRegistryEntry? GetModelEntry(string modelName)
        {
            return _entries.TryGetValue(modelName, out var entry) ? entry : null;
        }

        /// <summary>
        /// Checks if a model is registered.
        /// </summary>
        /// <param name="modelName">Name of the model file</param>
        /// <returns>True if the model is in the registry</returns>
        public bool IsModelRegistered(string modelName)
        {
            return _entries.ContainsKey(modelName);
        }

        /// <summary>
        /// Gets all registered model names.
        /// </summary>
        /// <returns>Collection of registered model names</returns>
        public IReadOnlyCollection<string> GetRegisteredModels()
        {
            return _entries.Keys;
        }

        private Dictionary<string, ModelRegistryEntry> LoadRegistry(string path)
        {
            if (!File.Exists(path))
            {
                _logger.LogWarning("Model registry file not found at '{Path}'. Using empty registry.", path);
                return new Dictionary<string, ModelRegistryEntry>(StringComparer.OrdinalIgnoreCase);
            }

            try
            {
                var json = File.ReadAllText(path);
                var registry = JsonSerializer.Deserialize<ModelRegistryData>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                if (registry?.Models == null)
                {
                    _logger.LogWarning("Model registry file is empty or invalid. Using empty registry.");
                    return new Dictionary<string, ModelRegistryEntry>(StringComparer.OrdinalIgnoreCase);
                }

                var entries = new Dictionary<string, ModelRegistryEntry>(StringComparer.OrdinalIgnoreCase);
                foreach (var entry in registry.Models)
                {
                    if (string.IsNullOrWhiteSpace(entry.Name) || entry.Urls == null || entry.Urls.Length == 0)
                    {
                        _logger.LogWarning("Skipping invalid registry entry: missing name or URLs");
                        continue;
                    }

                    entries[entry.Name] = entry;
                }

                _logger.LogInformation("Loaded {Count} models from registry", entries.Count);
                return entries;
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "Failed to parse model registry JSON");
                return new Dictionary<string, ModelRegistryEntry>(StringComparer.OrdinalIgnoreCase);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load model registry from '{Path}'", path);
                return new Dictionary<string, ModelRegistryEntry>(StringComparer.OrdinalIgnoreCase);
            }
        }

        /// <summary>
        /// Creates a default registry file with common models if it doesn't exist.
        /// </summary>
        /// <param name="path">Path to create the registry file</param>
        public static void CreateDefaultRegistry(string? path = null)
        {
            var registryPath = path ?? DefaultRegistryPath;
            if (File.Exists(registryPath))
            {
                return;
            }

            var directory = Path.GetDirectoryName(registryPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var defaultRegistry = new ModelRegistryData
            {
                Models = new[]
                {
                    new ModelRegistryEntry
                    {
                        Name = "yolo11n.onnx",
                        Description = "YOLO11 Nano object detection model",
                        Source = "Ultralytics",
                        Urls = new[]
                        {
                            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
                            "https://huggingface.co/ultralytics/YOLO11/resolve/main/yolo11n.onnx"
                        }
                    },
                    new ModelRegistryEntry
                    {
                        Name = "yolo11s.onnx",
                        Description = "YOLO11 Small object detection model",
                        Source = "Ultralytics",
                        Urls = new[]
                        {
                            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.onnx",
                            "https://huggingface.co/ultralytics/YOLO11/resolve/main/yolo11s.onnx"
                        }
                    },
                    new ModelRegistryEntry
                    {
                        Name = "yolo11m.onnx",
                        Description = "YOLO11 Medium object detection model",
                        Source = "Ultralytics",
                        Urls = new[]
                        {
                            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.onnx",
                            "https://huggingface.co/ultralytics/YOLO11/resolve/main/yolo11m.onnx"
                        }
                    },
                    new ModelRegistryEntry
                    {
                        Name = "yolov8n.onnx",
                        Description = "YOLOv8 Nano object detection model",
                        Source = "Ultralytics",
                        Urls = new[]
                        {
                            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
                            "https://huggingface.co/ultralytics/YOLOv8/resolve/main/yolov8n.onnx"
                        }
                    },
                    new ModelRegistryEntry
                    {
                        Name = "yolov8n-face.onnx",
                        Description = "YOLOv8 Nano single-class face detection model",
                        Source = "akanametov/yolo-face",
                        Urls = new[]
                        {
                            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx"
                        }
                    },
                    new ModelRegistryEntry
                    {
                        Name = "arcface-recognition.onnx",
                        Description = "ArcFace ResNet50 face recognition (insightface buffalo_l w600k_r50), 512-dim embeddings",
                        Source = "InsightFace via HuggingFace mirror richarrrddd/w600k_r50_v1",
                        Urls =
                        [
                            "https://huggingface.co/richarrrddd/w600k_r50_v1/resolve/main/w600k_r50.onnx"
                        ]
                    }
                }
            };

            var json = JsonSerializer.Serialize(defaultRegistry, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            File.WriteAllText(registryPath, json);
        }
    }

    /// <summary>
    /// Data class for the registry JSON structure.
    /// </summary>
    public class ModelRegistryData
    {
        public ModelRegistryEntry[] Models { get; set; } = Array.Empty<ModelRegistryEntry>();
    }

    /// <summary>
    /// Entry in the model registry.
    /// </summary>
    public class ModelRegistryEntry
    {
        /// <summary>
        /// Name of the model file (e.g., "yolo11n.onnx").
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Human-readable description of the model.
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Source organization or author of the model.
        /// </summary>
        public string Source { get; set; } = string.Empty;

        /// <summary>
        /// URLs to download the model from, in priority order.
        /// </summary>
        public string[] Urls { get; set; } = Array.Empty<string>();

        /// <summary>
        /// Optional expected SHA-256 of the model file, as 64 lowercase hex characters.
        /// When present, <see cref="DownloadingModelProvider"/> verifies the downloaded
        /// file against it and rejects mismatches; when absent, the download is accepted
        /// but logged as unverified.
        /// </summary>
        public string? Sha256 { get; set; }
    }
}
