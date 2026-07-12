package com.soumya.lore.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.soumya.lore.ui.components.LoadMoreRow
import com.soumya.lore.ui.components.LoreSearchField
import com.soumya.lore.ui.components.RecentSearchChip
import com.soumya.lore.ui.theme.LoreOutline
import com.soumya.lore.ui.theme.LoreTheme
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

private const val RECENT_SEARCHES_PAGE_SIZE = 3
private val RECENT_SEARCHES_CHIP_HEIGHT = 36.dp
private val RECENT_SEARCHES_CHIP_SPACING = 8.dp

/**
 * Entry screen. Owns the search-field text as local state — nothing else
 * in the app needs it, so hoisting it further up would add indirection
 * with no benefit. Voice input state lives in [HomeViewModel] since it
 * involves recording + a network call that should survive recomposition.
 */
@Composable
fun HomeScreen(
    onSearch: (query: String) -> Unit,
    modifier: Modifier = Modifier,
    viewModel: HomeViewModel = viewModel()
) {
    var query by remember { mutableStateOf("") }
    var visibleRecentCount by remember { mutableIntStateOf(RECENT_SEARCHES_PAGE_SIZE) }
    val voiceState by viewModel.voiceState.collectAsStateWithLifecycle()
    val waveformLevels by viewModel.waveformLevels.collectAsStateWithLifecycle()
    val recentSearches by viewModel.recentSearches.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Stage 1 ("query detachment"): search field shrinks and fades while the
    // query text detaches into a floating capsule that lifts slightly, then
    // the whole screen fades — before handing off to the Loading screen,
    // which continues the capsule's morph into a graph node.
    var isTransitioning by remember { mutableStateOf(false) }
    val fieldScale = remember { Animatable(1f) }
    val fieldAlpha = remember { Animatable(1f) }
    val capsuleAlpha = remember { Animatable(0f) }
    val capsuleOffsetY = remember { Animatable(0f) }
    val screenAlpha = remember { Animatable(1f) }

    fun triggerSearch(submittedQuery: String) {
        if (isTransitioning || submittedQuery.isBlank()) return
        isTransitioning = true
        viewModel.recordSearch(submittedQuery)
        scope.launch {
            coroutineScope {
                launch { fieldScale.animateTo(0.95f, tween(350, easing = FastOutSlowInEasing)) }
                launch { fieldAlpha.animateTo(0f, tween(300)) }
                launch { capsuleAlpha.animateTo(1f, tween(300)) }
                launch { capsuleOffsetY.animateTo(-20f, tween(350, easing = FastOutSlowInEasing)) }
            }
            screenAlpha.animateTo(0f, tween(180))
            onSearch(submittedQuery)
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) viewModel.onMicPressed() else viewModel.onPermissionDenied()
    }

    LaunchedEffect(voiceState) {
        when (val state = voiceState) {
            is VoiceState.Transcribed -> {
                query = state.transcript
                viewModel.consumeTranscript()
            }
            is VoiceState.Error -> {
                snackbarHostState.showSnackbar(state.message)
                viewModel.dismissError()
            }
            else -> Unit
        }
    }

    Scaffold(
        modifier = modifier,
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 24.dp)
                .graphicsLayer { alpha = screenAlpha.value },
        ) {
            // --- Header ---
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "LORE",
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 4.sp,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "Search your personal memory",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }

            // --- Main search area (fills remaining space, centered) ---
            // Uses explicit weighted spacers (not Arrangement.Center) so the
            // vertical bias is a direct, tunable number — Arrangement.Center
            // alone centers within this Column's own allotted space, which
            // sits slightly above true screen-center since the recent
            // searches block below is still taller than the header above.
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.weight(0.58f))
                Box(contentAlignment = Alignment.Center) {
                    LoreSearchField(
                        value = query,
                        onValueChange = { query = it },
                        onSearch = { triggerSearch(query) },
                        onMicClick = {
                            val hasPermission = ContextCompat.checkSelfPermission(
                                context,
                                Manifest.permission.RECORD_AUDIO
                            ) == PackageManager.PERMISSION_GRANTED

                            if (hasPermission) {
                                viewModel.onMicPressed()
                            } else {
                                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            }
                        },
                        isRecording = voiceState is VoiceState.Recording,
                        isTranscribing = voiceState is VoiceState.Transcribing,
                        waveformLevels = waveformLevels,
                        modifier = Modifier.graphicsLayer {
                            scaleX = fieldScale.value
                            scaleY = fieldScale.value
                            alpha = fieldAlpha.value
                        }
                    )

                    if (capsuleAlpha.value > 0f) {
                        Surface(
                            shape = RoundedCornerShape(50),
                            color = MaterialTheme.colorScheme.surface,
                            border = BorderStroke(1.dp, LoreOutline),
                            modifier = Modifier.graphicsLayer {
                                alpha = capsuleAlpha.value
                                translationY = capsuleOffsetY.value.dp.toPx()
                            }
                        ) {
                            Text(
                                text = query,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface,
                                maxLines = 1,
                                modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp)
                            )
                        }
                    }
                }

                Button(
                    onClick = { triggerSearch(query) },
                    enabled = !isTransitioning,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary
                    ),
                    modifier = Modifier.padding(top = 16.dp)
                ) {
                    Text("Search")
                }
                Spacer(modifier = Modifier.weight(0.42f))
            }

            // --- Recent searches + footer ---
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 20.dp)
            ) {
                Text(
                    text = "Recent Searches",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(
                            RECENT_SEARCHES_CHIP_HEIGHT * RECENT_SEARCHES_PAGE_SIZE +
                                RECENT_SEARCHES_CHIP_SPACING * (RECENT_SEARCHES_PAGE_SIZE - 1)
                        ),
                    verticalArrangement = Arrangement.spacedBy(RECENT_SEARCHES_CHIP_SPACING)
                ) {
                    items(recentSearches.take(visibleRecentCount)) { search ->
                        RecentSearchChip(
                            label = search,
                            onClick = {
                                query = search
                                triggerSearch(search)
                            }
                        )
                    }
                    if (visibleRecentCount < recentSearches.size) {
                        item {
                            LoadMoreRow(
                                onClick = {
                                    visibleRecentCount = (visibleRecentCount + RECENT_SEARCHES_PAGE_SIZE)
                                        .coerceAtMost(recentSearches.size)
                                }
                            )
                        }
                    }
                }

                Text(
                    text = "Private by Architecture",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 10.dp)
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun HomeScreenPreview() {
    LoreTheme {
        HomeScreen(onSearch = {})
    }
}
