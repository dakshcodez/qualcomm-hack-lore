package com.soumya.lore.ui.screens

import androidx.lifecycle.ViewModel
import com.soumya.lore.data.AnswerResult
import com.soumya.lore.network.LoreApiService
import com.soumya.lore.network.QueryOutcome
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Everything ResultScreen needs to know about the current /query call. */
sealed class QueryState {
    data object Idle : QueryState()
    data object Loading : QueryState()
    data class Success(val result: AnswerResult) : QueryState()
    data class Error(val message: String) : QueryState()
}

/**
 * Owns the /query round-trip so LoadingScreen and ResultScreen share one
 * result instead of each fetching independently. Scoped to the Activity
 * (created once in AppNavigation, above the NavHost) so it survives the
 * Loading -> Result navigation.
 */
class QueryViewModel : ViewModel() {

    private val _state = MutableStateFlow<QueryState>(QueryState.Idle)
    val state: StateFlow<QueryState> = _state.asStateFlow()

    /**
     * Runs the /query call and updates [state] with the outcome. This is a
     * suspend function (not fire-and-forget) so LoadingScreen can await it
     * directly inside its own animation `coroutineScope`, the same way it
     * previously awaited the mock delay.
     */
    suspend fun runQuery(text: String) {
        _state.value = QueryState.Loading
        _state.value = when (val outcome = LoreApiService.query(text)) {
            is QueryOutcome.Success -> QueryState.Success(outcome.result)
            is QueryOutcome.Error -> QueryState.Error(outcome.message)
        }
    }
}
