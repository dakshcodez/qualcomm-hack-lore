package com.soumya.lore.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONException

private const val PREFS_NAME = "lore_recent_searches"
private const val QUERIES_KEY = "queries"

/**
 * Persists the user's submitted search queries locally (SharedPreferences,
 * JSON-encoded) — nothing about search history ever leaves the device.
 * Pure storage: ordering, deduping, and capping the list is HomeViewModel's
 * job, this just round-trips whatever list it's given.
 */
class RecentSearchesStore(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): List<String> {
        val raw = prefs.getString(QUERIES_KEY, null) ?: return emptyList()
        return try {
            val array = JSONArray(raw)
            List(array.length()) { array.getString(it) }
        } catch (_: JSONException) {
            emptyList()
        }
    }

    fun save(queries: List<String>) {
        prefs.edit().putString(QUERIES_KEY, JSONArray(queries).toString()).apply()
    }
}
