package com.soumya.lore.navigation

import android.net.Uri
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.soumya.lore.ui.screens.HomeScreen
import com.soumya.lore.ui.screens.LoadingScreen
import com.soumya.lore.ui.screens.QueryViewModel
import com.soumya.lore.ui.screens.ResultScreen

private const val QUERY_ARG = "query"
private const val HOME_ROUTE = "home"
private const val LOADING_ROUTE = "loading/{$QUERY_ARG}"
private const val RESULT_ROUTE = "result/{$QUERY_ARG}"

/**
 * Single source of truth for how screens connect: Home -> Loading -> Result.
 * The query text is passed forward as a (URL-encoded) nav argument, since
 * Home only ever produces a string. The actual /query result is different:
 * it's fetched once, in Loading, and needs to reach Result unchanged, so it
 * lives in a single [QueryViewModel] instance created here — above the
 * NavHost, so it's scoped to this composable's caller (the Activity) rather
 * than to any one destination, and is shared by both Loading and Result.
 */
@Composable
fun AppNavigation() {
    val navController = rememberNavController()
    val queryViewModel: QueryViewModel = viewModel()

    NavHost(navController = navController, startDestination = HOME_ROUTE) {
        composable(HOME_ROUTE) {
            HomeScreen(
                onSearch = { query ->
                    navController.navigate("loading/${Uri.encode(query)}")
                }
            )
        }

        composable(
            route = LOADING_ROUTE,
            arguments = listOf(navArgument(QUERY_ARG) { type = NavType.StringType })
        ) { backStackEntry ->
            val query = backStackEntry.arguments?.getString(QUERY_ARG).orEmpty()
            LoadingScreen(
                query = query,
                queryViewModel = queryViewModel,
                onComplete = {
                    navController.navigate("result/${Uri.encode(query)}") {
                        // Replace Loading in the back stack so the user can't
                        // navigate "back" into a finished loading screen.
                        popUpTo(HOME_ROUTE)
                    }
                }
            )
        }

        composable(
            route = RESULT_ROUTE,
            arguments = listOf(navArgument(QUERY_ARG) { type = NavType.StringType }),
            // Fast, minimal fade rather than the default slide/crossfade —
            // Loading's own node fades out right before this fires, so a
            // quick fade reads as a continuation rather than a hard cut.
            enterTransition = { fadeIn(tween(120)) },
            exitTransition = { fadeOut(tween(80)) }
        ) { backStackEntry ->
            val query = backStackEntry.arguments?.getString(QUERY_ARG).orEmpty()
            ResultScreen(
                query = query,
                queryViewModel = queryViewModel,
                onBack = { navController.popBackStack() },
                onNewSearch = { navController.popBackStack() }
            )
        }
    }
}
