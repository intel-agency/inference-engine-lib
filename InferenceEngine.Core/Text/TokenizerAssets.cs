using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging;
using Microsoft.ML.Tokenizers;

namespace InferenceEngine.Core.Text;

/// <summary>
/// Resolves and constructs the <see cref="Tokenizer"/> instances used by the text engines.
/// Downloads the vocab/model asset files once (delegating to <see cref="DownloadingModelProvider"/>
/// for cache + retry), then builds the configured tokenizer over a read-only file stream.
/// </summary>
internal static class TokenizerAssets
{
    /// <summary>MiniLM (all-MiniLM-L6-v2) BERT WordPiece vocabulary file.</summary>
    internal const string MiniLmVocabFileName = "vocab.txt";

    /// <summary>MiniLM vocab download URL (uncased BERT WordPiece).</summary>
    internal const string MiniLmVocabUrl =
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/vocab.txt";

    /// <summary>Phi-3 SentencePiece tokenizer model file.</summary>
    internal const string Phi3TokenizerFileName = "tokenizer.model";

    /// <summary>Phi-3 SentencePiece tokenizer model download URL.</summary>
    internal const string Phi3TokenizerUrl =
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32/tokenizer.model";

    /// <summary>
    /// Loads the all-MiniLM-L6-v2 tokenizer (an uncased BERT WordPiece model) from a cached or
    /// freshly-downloaded <c>vocab.txt</c>. Returns the configured <see cref="BertTokenizer"/>.
    /// </summary>
    /// <param name="options">Engine options; the cache root is derived from the default model cache.</param>
    /// <param name="logger">Logger for download/failure diagnostics.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A <see cref="Tokenizer"/> for all-MiniLM-L6-v2.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the vocab asset cannot be fetched.</exception>
    public static async Task<Tokenizer> LoadMiniLmTokenizerAsync(
        InferenceEngineOptions options,
        ILogger logger,
        CancellationToken ct = default)
    {
        var vocabPath = await DownloadAssetAsync(
            MiniLmVocabFileName,
            new[] { MiniLmVocabUrl },
            options,
            logger,
            ct).ConfigureAwait(false);

        var bertOptions = new BertOptions { LowerCaseBeforeTokenization = true };
        await using var stream = new FileStream(vocabPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var tokenizer = BertTokenizer.Create(stream, bertOptions);
        logger.LogInformation("Loaded MiniLM (all-MiniLM-L6-v2) BERT tokenizer from {Path}", vocabPath);
        return tokenizer;
    }

    /// <summary>
    /// Loads the Phi-3 SentencePiece tokenizer from a cached or freshly-downloaded
    /// <c>tokenizer.model</c>. No BOS/EOS is auto-inserted; the engine drives EOS via
    /// <see cref="StreamingOptions.EosTokenIds"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In <c>Microsoft.ML.Tokenizers</c> 1.0.3 the <c>SentencePieceTokenizer</c> type has no public
    /// factory of its own. The public factory that reads a SentencePiece model and returns a
    /// <see cref="SentencePieceTokenizer"/>-derived instance is <see cref="LlamaTokenizer.Create"/>,
    /// whose signature matches the intended SentencePiece usage. Phi-3's tokenizer.model is a
    /// SentencePiece model and is read here via that factory.
    /// </para>
    /// </remarks>
    /// <param name="options">Engine options; the cache root is derived from the default model cache.</param>
    /// <param name="logger">Logger for download/failure diagnostics.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A <see cref="Tokenizer"/> for Phi-3-mini-4k-instruct.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the model asset cannot be fetched.</exception>
    public static async Task<Tokenizer> LoadPhi3TokenizerAsync(
        InferenceEngineOptions options,
        ILogger logger,
        CancellationToken ct = default)
    {
        var modelPath = await DownloadAssetAsync(
            Phi3TokenizerFileName,
            new[] { Phi3TokenizerUrl },
            options,
            logger,
            ct).ConfigureAwait(false);

        var specialTokens = new System.Collections.Generic.Dictionary<string, int>
        {
            ["<|endoftext|>"] = 32000,
            ["<|im_start|>"] = 32001,
            ["<|im_end|>"] = 32002,
        };

        await using var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        // LlamaTokenizer is the public SentencePiece-backed factory in Tokenizers 1.0.3 and returns
        // a SentencePieceTokenizer-derived Tokenizer. addBos/addEos=false: the engine owns EOS.
        var tokenizer = LlamaTokenizer.Create(
            stream,
            addBeginOfSentence: false,
            addEndOfSentence: false,
            specialTokens);
        logger.LogInformation("Loaded Phi-3 SentencePiece tokenizer from {Path}", modelPath);
        return tokenizer;
    }

    /// <summary>
    /// Downloads (or reuses a cached copy of) a single tokenizer asset file under the engine's
    /// tokenizers cache directory using <see cref="DownloadingModelProvider"/>.
    /// </summary>
    private static async Task<string> DownloadAssetAsync(
        string fileName,
        string[] urls,
        InferenceEngineOptions options,
        ILogger logger,
        CancellationToken ct)
    {
        _ = options ?? throw new ArgumentNullException(nameof(options));
        string cacheDir = GetTokenizersCacheDirectory();
        Directory.CreateDirectory(cacheDir);

        var provider = new DownloadingModelProvider(
            modelName: fileName,
            downloadUrls: urls,
            cacheDirectory: cacheDir,
            logger: logger);

        try
        {
            return await provider.GetModelPathAsync(ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to obtain tokenizer asset '{File}' from {UrlCount} source(s)", fileName, urls.Length);
            throw new InvalidOperationException(
                $"Tokenizer asset '{fileName}' could not be loaded or downloaded. " +
                "Provide network access or place the file in the tokenizers cache directory.", ex);
        }
    }

    /// <summary>Tokenizers cache directory, located alongside the default model cache.</summary>
    private static string GetTokenizersCacheDirectory() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "InferenceEngine",
        "Tokenizers");
}
