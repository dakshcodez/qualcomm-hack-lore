package com.soumya.lore.ui.components

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.soumya.lore.ui.theme.LoreOutline

private val FIELD_SHAPE = RoundedCornerShape(20.dp)
private val FIELD_HEIGHT = 64.dp
private val WAVEFORM_BAR_MIN_HEIGHT = 4.dp
private val WAVEFORM_BAR_MAX_HEIGHT = 52.dp

/**
 * The primary search field for Lore. Deliberately larger and more rounded
 * than a default text field so it reads as the "hero" element on Home.
 *
 * The mic button is intentionally small and neutral-colored at rest — it
 * should never compete visually with search itself (see design brief:
 * "search product first, voice product second"). While recording, the text
 * row is replaced by a live waveform (same field shape/border, emerald —
 * matching the design system's rule that emerald marks active state) and
 * the mic becomes a square stop button, mirroring ChatGPT's voice input.
 */
@Composable
fun LoreSearchField(
    value: String,
    onValueChange: (String) -> Unit,
    onSearch: () -> Unit,
    onMicClick: () -> Unit,
    modifier: Modifier = Modifier,
    isRecording: Boolean = false,
    isTranscribing: Boolean = false,
    waveformLevels: List<Float> = emptyList()
) {
    if (isRecording) {
        RecordingRow(waveformLevels = waveformLevels, onMicClick = onMicClick, modifier = modifier)
        return
    }

    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier
            .fillMaxWidth()
            .height(FIELD_HEIGHT)
            .background(MaterialTheme.colorScheme.surface, FIELD_SHAPE),
        placeholder = {
            Text(
                text = "Ask Lore anything...",
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        },
        leadingIcon = {
            Icon(
                imageVector = Icons.Default.Search,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        },
        trailingIcon = {
            IconButton(onClick = onMicClick, enabled = !isTranscribing) {
                if (isTranscribing) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.Mic,
                        contentDescription = "Voice search",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        },
        singleLine = true,
        shape = FIELD_SHAPE,
        colors = OutlinedTextFieldDefaults.colors(
            focusedContainerColor = MaterialTheme.colorScheme.surface,
            unfocusedContainerColor = MaterialTheme.colorScheme.surface,
            focusedBorderColor = MaterialTheme.colorScheme.primary,
            unfocusedBorderColor = LoreOutline
        ),
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
        keyboardActions = KeyboardActions(onSearch = { onSearch() })
    )
}

/**
 * Recording state: same outer shape/height as the idle field (so the swap
 * doesn't jump), an emerald border marking it active, a live bar waveform
 * in place of typed text, and a square stop button in place of the mic.
 */
@Composable
private fun RecordingRow(
    waveformLevels: List<Float>,
    onMicClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(FIELD_HEIGHT)
            .background(MaterialTheme.colorScheme.surface, FIELD_SHAPE)
            .border(1.dp, MaterialTheme.colorScheme.primary, FIELD_SHAPE)
            .padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Default.Search,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant
        )
        WaveformBars(
            levels = waveformLevels,
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight()
                .padding(horizontal = 14.dp)
        )
        IconButton(onClick = onMicClick) {
            Icon(
                imageVector = Icons.Default.Stop,
                contentDescription = "Stop recording",
                tint = MaterialTheme.colorScheme.primary
            )
        }
    }
}

/**
 * Live equalizer-style waveform: a fixed number of bars whose heights track
 * [levels] (oldest first, most recent last), each animating smoothly toward
 * its new target so updates read as motion rather than a jump-cut per frame.
 */
@Composable
private fun WaveformBars(levels: List<Float>, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(3.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        levels.forEach { level ->
            val barHeight by animateDpAsState(
                targetValue = WAVEFORM_BAR_MIN_HEIGHT +
                    (WAVEFORM_BAR_MAX_HEIGHT - WAVEFORM_BAR_MIN_HEIGHT) * level.coerceIn(0f, 1f),
                animationSpec = tween(120),
                label = "waveformBar"
            )
            Box(
                modifier = Modifier
                    .width(3.dp)
                    .height(barHeight)
                    .background(MaterialTheme.colorScheme.primary, RoundedCornerShape(2.dp))
            )
        }
    }
}
