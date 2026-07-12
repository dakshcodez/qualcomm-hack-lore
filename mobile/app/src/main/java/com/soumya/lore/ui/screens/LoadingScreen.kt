package com.soumya.lore.ui.screens

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.soumya.lore.data.generateKnowledgeGraph
import com.soumya.lore.data.pathPosition
import com.soumya.lore.ui.components.KnowledgeGraphCanvas
import com.soumya.lore.ui.theme.LoreTheme
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
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
private const val LOADING_PHRASE_INTERVAL_MS = 2400L

private val LoadingPhrases = listOf(
    "Exploring your digital mind",
    "Diving deep into your memories",
    "Piecing together the lore"
)

/**
 * Three-stage retrieval visualization: the query detaches from Home and
 * enters as a node (continuation of Home's own detach animation), travels
 * a knowledge graph while a few nodes light up as relevant matches, then
 * the node grows into a soft glow and fades out, handing off directly to
 * Results with no intermediate card silhouette.
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
    // Pixel-space translation applied on top of the capsule's normal layout
    // position. The actual target is computed from real measurements (see
    // below) rather than guessed, so this starts at zero and only animates
    // once the true delta to the dot's starting point is known.
    val capsuleTranslate = remember { Animatable(Offset.Zero, Offset.VectorConverter) }
    val capsuleScale = remember { Animatable(1f) }
    val queryAlpha = remember { Animatable(0f) }
    val queryRadius = remember { Animatable(4.5f) }
    val lapProgress = remember { Animatable(0f) }
    var isTraveling by remember { mutableStateOf(false) }
    // The dot no longer flies in on its own — it simply appears at the
    // path's start once the capsule arrives there, so it reads as the
    // capsule handing off to it rather than two things converging.
    val queryPosition = remember {
        Animatable(graph.travelWaypoints.first(), Offset.VectorConverter)
    }

    // Real measurements, captured via onGloballyPositioned below, so the
    // capsule's flight target is the dot's *actual* on-screen starting
    // point — not a guessed nudge.
    var graphBoxPositionPx by remember { mutableStateOf<Offset?>(null) }
    var graphBoxSizePx by remember { mutableStateOf<IntSize?>(null) }
    var capsulePositionPx by remember { mutableStateOf<Offset?>(null) }

    // Bottom pulsing status text: cycles through LoadingPhrases on a timer,
    // with a continuous breathing alpha independent of that cycle — the
    // composable is torn down (cancelling both) the moment onComplete()
    // navigates away, so neither needs explicit cleanup.
    var loadingPhraseIndex by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        // Advances through LoadingPhrases once, in order, then stops on the
        // last one — no wrap-around back to the first.
        while (loadingPhraseIndex < LoadingPhrases.size - 1) {
            delay(LOADING_PHRASE_INTERVAL_MS)
            loadingPhraseIndex++
        }
    }
    val loadingTextAlpha by rememberInfiniteTransition(label = "loadingTextPulse").animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(900, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "loadingTextPulseAlpha"
    )

    LaunchedEffect(query) {
        // Wait for the first real layout pass so the geometry below is accurate.
        snapshotFlow { Triple(graphBoxPositionPx, graphBoxSizePx, capsulePositionPx) }
            .first { (pos, size, capsulePos) -> pos != null && size != null && capsulePos != null }

        val graphOrigin = graphBoxPositionPx!!
        val graphSize = graphBoxSizePx!!
        val startFraction = graph.travelWaypoints.first()
        val targetPointInRoot = Offset(
            graphOrigin.x + startFraction.x * graphSize.width,
            graphOrigin.y + startFraction.y * graphSize.height
        )
        val flightDelta = targetPointInRoot - capsulePositionPx!!

        // Stage 1 continuation: the capsule that arrived from Home flies
        // fast toward the dot's actual starting point, shrinking as it
        // goes, and disappears there — replaced by the glowing query node,
        // rather than crossfading in place.
        coroutineScope {
            launch { capsuleTranslate.animateTo(flightDelta, tween(300, easing = FastOutSlowInEasing)) }
            launch { capsuleScale.animateTo(0.15f, tween(300, easing = FastOutSlowInEasing)) }
            launch { capsuleAlpha.animateTo(0f, tween(300, easing = FastOutSlowInEasing)) }
            launch { queryAlpha.animateTo(1f, tween(220, delayMillis = 160, easing = FastOutSlowInEasing)) }
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
        // into a glow, and fade the node itself out — navigating straight
        // to Results right after, with no intermediate card silhouette.
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
        queryAlpha.animateTo(0f, tween(220))

        onComplete()
    }

    Scaffold(modifier = modifier) { innerPadding ->
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding)
            .background(GraphBackground)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 32.dp, start = 24.dp, end = 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "LORE",
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 4.sp,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }

            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxSize()
                    .onGloballyPositioned {
                        graphBoxPositionPx = it.positionInRoot()
                        graphBoxSizePx = it.size
                    }
            ) {
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
                        // Spawns near screen-center, mirroring where the
                        // capsule sat on Home's search field — deliberately
                        // NOT near travelWaypoints.first() (which sits close
                        // to the left edge), so there's real distance to fly.
                        // Shifted up 100dp from true center per request.
                        .align(Alignment.Center)
                        .offset(y = (-100).dp)
                        .onGloballyPositioned { capsulePositionPx = it.positionInRoot() }
                        .graphicsLayer {
                            translationX = capsuleTranslate.value.x
                            translationY = capsuleTranslate.value.y
                            scaleX = capsuleScale.value
                            scaleY = capsuleScale.value
                            alpha = capsuleAlpha.value
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

        Text(
            text = LoadingPhrases[loadingPhraseIndex],
            style = MaterialTheme.typography.titleMedium,
            color = Color(0xFF9B9B9B),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 100.dp)
                .graphicsLayer { alpha = loadingTextAlpha }
        )
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
