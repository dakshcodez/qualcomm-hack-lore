package com.soumya.lore.audio

import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import java.io.File
import java.io.IOException

/**
 * Thin wrapper around [MediaRecorder]. Knows nothing about networking or UI —
 * it just produces a `.m4a` file on disk that the caller is responsible for
 * uploading and deleting.
 *
 * MediaRecorder (not AudioRecord) is deliberate here: it encodes straight to
 * a compressed, Sarvam-supported format (AAC/M4A) with a few lines of setup.
 * AudioRecord would hand back raw PCM and require writing a WAV header or a
 * manual encoder — only worth it for real-time streaming, which this isn't.
 */
class AudioRecorder(private val context: Context) {

    private var recorder: MediaRecorder? = null
    private var outputFile: File? = null

    /** Starts recording to a new cache file. Throws [IOException] on failure. */
    @Throws(IOException::class)
    fun start(): File {
        val file = File(context.cacheDir, "lore_query_${System.currentTimeMillis()}.m4a")

        val mediaRecorder = newMediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setAudioSamplingRate(16_000) // Sarvam performs best at 16kHz
            setAudioEncodingBitRate(96_000)
            setMaxDuration(MAX_RECORDING_DURATION_MS) // matches Sarvam's 30s REST limit
            setOutputFile(file.absolutePath)
            prepare()
            start()
        }

        recorder = mediaRecorder
        outputFile = file
        return file
    }

    /** Stops the current recording and returns the recorded file, or null if nothing usable was captured. */
    fun stop(): File? {
        val current = recorder ?: return null
        recorder = null
        return try {
            current.stop()
            current.release()
            outputFile
        } catch (e: RuntimeException) {
            // Thrown when stop() is called too soon after start() with no audio captured.
            current.release()
            outputFile?.delete()
            null
        } finally {
            outputFile = null
        }
    }

    /**
     * Peak amplitude (0..32767 roughly) since the last call — MediaRecorder's
     * own running max, reset on each read. Returns 0 when not recording.
     * Used to drive the live waveform; not persisted anywhere.
     */
    fun currentAmplitude(): Int = try {
        recorder?.maxAmplitude ?: 0
    } catch (_: IllegalStateException) {
        0
    }

    /** Cancels an in-progress recording without producing a result (e.g. user backs out). */
    fun cancel() {
        try {
            recorder?.stop()
        } catch (_: RuntimeException) {
            // No data was captured — nothing to clean up beyond releasing/deleting below.
        }
        recorder?.release()
        recorder = null
        outputFile?.delete()
        outputFile = null
    }

    private fun newMediaRecorder(): MediaRecorder =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(context)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }

    companion object {
        const val MAX_RECORDING_DURATION_MS = 30_000
    }
}
