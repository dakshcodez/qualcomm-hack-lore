package com.soumya.lore.network

import com.soumya.lore.BuildConfig
import com.soumya.lore.data.AnswerResult
import com.soumya.lore.data.Source
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

/** Outcome of a /query call — the network layer never leaks raw JSON past this. */
sealed class QueryOutcome {
    data class Success(val result: AnswerResult) : QueryOutcome()
    data class Error(val message: String) : QueryOutcome()
}

/**
 * Talks to the PC's local FastAPI backend (pc/api/routes_query.py's
 * POST /query — see pc/api/schemas.py for the exact request/response
 * shape this mirrors). BuildConfig.PC_BASE_URL is the PC's local IP,
 * configured via mobile/local.properties per CLAUDE.md's hackathon
 * networking notes (same mechanism as SARVAM_API_KEY).
 */
object LoreApiService {

    suspend fun query(text: String, modality: String = "text"): QueryOutcome = withContext(Dispatchers.IO) {
        val requestJson = JSONObject()
            .put("text", text)
            .put("modality", modality)
        val requestBody = requestJson.toString().toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url("${BuildConfig.PC_BASE_URL.trimEnd('/')}/query")
            .post(requestBody)
            .build()

        try {
            NetworkModule.client.newCall(request).execute().use { response ->
                val bodyString = response.body?.string().orEmpty()

                if (!response.isSuccessful) {
                    return@withContext QueryOutcome.Error(errorMessageFor(response.code))
                }

                val json = JSONObject(bodyString)
                QueryOutcome.Success(
                    AnswerResult(
                        answer = json.optString("answer"),
                        sources = parseSources(json.optJSONArray("sources"))
                    )
                )
            }
        } catch (_: IOException) {
            QueryOutcome.Error("Couldn't reach your PC — check you're on the same WiFi.")
        } catch (_: org.json.JSONException) {
            QueryOutcome.Error("Got an unexpected response from your PC. Try again.")
        }
    }

    private fun parseSources(sourcesJson: org.json.JSONArray?): List<Source> {
        if (sourcesJson == null) return emptyList()
        return buildList {
            for (i in 0 until sourcesJson.length()) {
                val s = sourcesJson.getJSONObject(i)
                add(
                    Source(
                        title = s.optString("title"),
                        location = s.optString("location"),
                        excerpt = s.optString("excerpt"),
                        fileType = s.optString("file_type")
                    )
                )
            }
        }
    }

    private fun errorMessageFor(code: Int): String = when (code) {
        422 -> "That search didn't go through — try again."
        in 500..599 -> "Something went wrong on your PC. Check its logs."
        else -> "Couldn't get an answer — try again."
    }
}
