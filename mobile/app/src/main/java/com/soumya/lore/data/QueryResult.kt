package com.soumya.lore.data

/**
 * UI-facing shape of a single retrieved source. Field names match
 * pc/api/schemas.py's `SourceItem` exactly (title, location, excerpt,
 * file_type -> fileType) so LoreApiService can parse the real /query
 * response directly into this without any renaming/mapping layer.
 */
data class Source(
    val title: String,
    val location: String,
    val excerpt: String,
    val fileType: String
)

/** UI-facing shape of a full /query response — mirrors QueryResponse. */
data class AnswerResult(
    val answer: String,
    val sources: List<Source>
)
