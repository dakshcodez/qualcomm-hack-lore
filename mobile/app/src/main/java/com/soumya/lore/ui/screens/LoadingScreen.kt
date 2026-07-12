package com.soumya.lore.ui.screens

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.soumya.lore.data.generateKnowledgeGraph
import com.soumya.lore.data.pathPosition
import com.soumya.lore.ui.components.KnowledgeGraphCanvas
import com.soumya.lore.ui.theme.LoreOutline
import com.soumya.lore.ui.theme.LoreTheme
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

/** The graph canvas sits on a near-black background distinct from the rest of the app. */
private val GraphBackground = Color(0xFF0D0D0D)

// Stage 2 has no fixed duration and never loops. The dot advances toward
// (but never quite reaches) TRICKLE_CEILING in successive steps, each
// covering TRICKLE_APPROACH_FACTOR of the remaining distance — a classic
// "unknown total duration" progress pattern: naturally decelerating, and
// entirely paced by how long the backend actually takes to respond, since
// each step only fires after the previous one completes. The instant the
// backend responds, a single fast tween covers whatever distance remains.
private const val TRICKLE_CEILING = 0.92f
private const val TRICKLE_APPROACH_FACTOR = 0.35f
private const val TRICKLE_STEP_DURATION_MS = 900
private const val FAST_ARRIVAL_TWEEN_MS = 500
private const val LIGHT_UP_TWEEN_MS = 700

/**
 * Three-stage retrieval visualization: the query detaches from Home and
 * enters as a node (continuation of Home's own detach animation), travels
 * a knowledge graph while a few nodes light up as relevant matches, then
 * contracts back into a single node that expands into the shape of the
 * answer card right before navigating to Results.
 */
@Composable
fun LoadingScreen(
    query: String,
    onComplete: () -> Unit,
    queryViewModel: QueryViewModel,
    modifier: Modifier = Modifier
) {
    val graph = remember { generateKnowledgeGraph() }
    val emerald = MaterialTheme.colorScheme.primary

    val nodeGlow = remember { graph.nodes.map { Animatable(0.28f) } }
    // Starts invisible: the graph must not appear until the capsule has
    // fully morphed into a node. Also doubles as the Stage 3 fade-out.
    val graphVisibility = remember { Animatable(0f) }
    val capsuleAlpha = remember { Animatable(1f) }
    val queryAlpha = remember { Animatable(0f) }
    val queryRadius = remember { Animatable(4.5f) }
    val cardAlpha = remember { Animatable(0f) }
    val lapProgress = remember { Animatable(0f) }
    var isTraveling by remember { mutableStateOf(false) }
    val queryPosition = remember {
        Animatable(Offset(-0.12f, graph.travelWaypoints.first().y), Offset.VectorConverter)
    }

    LaunchedEffect(query) {
        // Stage 1 continuation: the capsule that arrived from Home morphs
        // smoothly (crossfade, not a hard cut) into the glowing query node.
        coroutineScope {
            launch { capsuleAlpha.animateTo(0f, tween(350, easing = FastOutSlowInEasing)) }
            launch { queryAlpha.animateTo(1f, tween(400, easing = FastOutSlowInEasing)) }
            launch {
                queryPosition.animateTo(
                    graph.travelWaypoints.first(),
                    tween(450, easing = FastOutSlowInEasing)
                )
            }
        }

        // Only now — after the node has fully formed — does the graph fade
        // in around it. Nothing about the graph renders before this point.
        graphVisibility.animateTo(1f, tween(450, easing = FastOutSlowInEasing))

        // Stage 2: the dot trickles forward — never resetting to the start —
        // pacing itself entirely off how long the real /query call actually
        // takes. Relevant nodes light up (once) as soon as the dot's real
        // progress reaches their position on the path.
        isTraveling = true
        val litNodeIds = mutableSetOf<Int>()
        fun lightDueNodes() {
            for ((nodeIndex, pathT) in graph.relevantNodePlan) {
                if (lapProgress.value >= pathT && litNodeIds.add(nodeIndex)) {
                    launch { nodeGlow[nodeIndex].animateTo(1f, tween(LIGHT_UP_TWEEN_MS, easing = FastOutSlowInEasing)) }
                }
            }
        }

        val trickleJob = launch {
            var target = 0f
            while (true) {
                target = (target + (TRICKLE_CEILING - target) * TRICKLE_APPROACH_FACTOR)
                    .coerceAtMost(TRICKLE_CEILING)
                lapProgress.animateTo(target, tween(TRICKLE_STEP_DURATION_MS, easing = LinearEasing))
                lightDueNodes()
            }
        }

        queryViewModel.runQuery(query)

        // Stage 3: the instant the backend responds, stop trickling and
        // dash straight to the true end of the path — then hand off to a
        // deliberate move toward center, fade the graph away, grow the node
        // into a glow, and crossfade into a static card silhouette before
        // navigating, so the real Results screen lands on an already-settled
        // shape instead of a mid-motion one.
        trickleJob.cancel()
        coroutineScope {
            launch { lapProgress.animateTo(1f, tween(FAST_ARRIVAL_TWEEN_MS, easing = FastOutSlowInEasing)) }
            // Anything not yet lit (dot hadn't geometrically reached it) lights now.
            for ((nodeIndex, _) in graph.relevantNodePlan) {
                if (litNodeIds.add(nodeIndex)) {
                    launch { nodeGlow[nodeIndex].animateTo(1f, tween(LIGHT_UP_TWEEN_MS, easing = FastOutSlowInEasing)) }
                }
            }
        }
        // Snap queryPosition to the true end before switching the canvas
        // over to following it, so there's no visible jump.
        queryPosition.snapTo(pathPosition(graph.travelWaypoints, lapProgress.value))
        isTraveling = false
        coroutineScope {
            launch { queryPosition.animateTo(Offset(0.5f, 0.42f), tween(500, easing = FastOutSlowInEasing)) }
            launch { graphVisibility.animateTo(0f, tween(400)) }
            launch { queryRadius.animateTo(170f, tween(500, easing = FastOutSlowInEasing)) }
            launch { nodeGlow.map { async { it.animateTo(0f, tween(400)) } }.awaitAll() }
        }
        coroutineScope {
            launch { queryAlpha.animateTo(0f, tween(220)) }
            launch { cardAlpha.animateTo(1f, tween(220)) }
        }

        onComplete()
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(GraphBackground)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 32.dp, start = 24.dp, end = 24.dp),
            ) {
                Text(
                    text = "LORE",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 3.sp,
                    color = Color(0xFFECECEC)
                )
                Text(
                    text = "Searching your memory...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFF9B9B9B),
                    modifier = Modifier.padding(top = 4.dp)
                )
            }

            Box(modifier = Modifier.weight(1f).fillMaxSize()) {
                KnowledgeGraphCanvas(
                    graph = graph,
                    nodeGlow = nodeGlow,
                    graphVisibility = graphVisibility,
                    queryNodeAlpha = queryAlpha,
                    queryNodePosition = queryPosition,
                    queryNodeRadius = queryRadius,
                    lapProgress = lapProgress,
                    isTraveling = isTraveling,
                    emeraldColor = emerald,
                    modifier = Modifier.fillMaxSize()
                )

                Surface(
                    color = MaterialTheme.colorScheme.surface,
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier
                        .align(Alignment.CenterStart)
                        .padding(start = 24.dp)
                        .graphicsLayer { alpha = capsuleAlpha.value }
                ) {
                    Text(
                        text = query,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        maxLines = 1,
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp)
                    )
                }

                Surface(
                    color = MaterialTheme.colorScheme.surface,
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, LoreOutline),
                    modifier = Modifier
                        .align(Alignment.Center)
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                        .height(140.dp)
                        .graphicsLayer { alpha = cardAlpha.value }
                ) {}
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun LoadingScreenPreview() {
    LoreTheme {
        LoadingScreen(query = "best hackathons", onComplete = {}, queryViewModel = QueryViewModel())
    }
}
