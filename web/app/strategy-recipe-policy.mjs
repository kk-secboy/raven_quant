export const STRATEGY_CREATION_RECIPE_IDS = Object.freeze([
  "short_relative_strength",
  "swing_trend",
  "long_quality_value",
  "index_enhancement",
  "full_market_multifactor",
]);

const strategyCreationRecipeIds = new Set(STRATEGY_CREATION_RECIPE_IDS);

/**
 * The API also returns offline/minute research recipes.  The strategy creation
 * UI is deliberately allow-listed so those recipes cannot silently become a
 * production strategy when the backend catalogue grows.
 *
 * @template {{ id: string }} T
 * @param {T[]} recipes
 * @returns {T[]}
 */
export function visibleStrategyCreationRecipes(recipes) {
  return recipes.filter((recipe) => strategyCreationRecipeIds.has(recipe.id));
}

/**
 * @param {{ id: string, config_overrides?: Record<string, unknown> } | undefined} recipe
 */
export function recipeUsesQlibBaseline(recipe) {
  return Boolean(
    recipe
    && strategyCreationRecipeIds.has(recipe.id)
    && recipe.config_overrides?.factor_source_mode === "qlib_baseline",
  );
}

/**
 * @param {{ id: string, config_overrides?: Record<string, unknown> } | undefined} recipe
 * @param {string} mode
 * @param {number} challengerWeight
 * @param {number} selectedFactorCount
 */
export function factorSourceSelectionIsValid(
  recipe,
  mode,
  challengerWeight,
  selectedFactorCount,
) {
  const qlibBaselineRecipe = recipeUsesQlibBaseline(recipe);
  const modeValid = (
    (!qlibBaselineRecipe && mode === "promoted_only")
    || (qlibBaselineRecipe && mode === "qlib_baseline")
    || (
      qlibBaselineRecipe
      && mode === "qlib_baseline_plus_challenger"
      && challengerWeight > 0
      && challengerWeight < 1
    )
    || (qlibBaselineRecipe && mode === "qlib_challenger_replacement")
  );
  return modeValid && (mode === "qlib_baseline" || selectedFactorCount > 0);
}
