package com.soumya.lore.data

/**
 * Preview-only stand-in for a real /query response — used by @Preview
 * composables so they render without a live PC connection. Real screens
 * get their AnswerResult from QueryViewModel/LoreApiService instead.
 */
fun mockAnswerFor(query: String): AnswerResult = AnswerResult(
    answer = "Based on your indexed files, here's what I found related to " +
        "\"$query\". This is placeholder text until the PC and Cloud AI 100 " +
        "pipeline is connected.",
    sources = listOf(
        Source(
            title = "Q3_Planning_Notes.docx",
            location = "/home/user/Documents/Q3_Planning_Notes.docx",
            excerpt = "...relevant excerpt mentioning \"$query\" would appear here...",
            fileType = "docx"
        ),
        Source(
            title = "research-paper.pdf",
            location = "/home/user/Documents/research-paper.pdf",
            excerpt = "...another matching excerpt would appear here...",
            fileType = "pdf"
        ),
        Source(
            title = "meeting-summary.txt",
            location = "/home/user/Documents/meeting-summary.txt",
            excerpt = "...a third matching excerpt would appear here...",
            fileType = "txt"
        )
    )
)
