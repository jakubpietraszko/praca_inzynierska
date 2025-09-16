import { createFeatureSelector, createSelector } from '@ngrx/store';
import { ConfigState, XAIMethodConfig } from './config.model';

export const selectConfigState = createFeatureSelector<ConfigState>('config');

export const selectConfig = createSelector(
  selectConfigState,
  (state: ConfigState) => state.config
);

export const selectConfigLoading = createSelector(
  selectConfigState,
  (state: ConfigState) => state.loading
);

export const selectConfigError = createSelector(
  selectConfigState,
  (state: ConfigState) => state.error
);

export const selectMethods = createSelector(
  selectConfig,
  (config) => config?.methods || {}
);

export const selectMethodNames = createSelector(
  selectMethods,
  (methods) => Object.keys(methods)
);

export const selectMethodConfig = (methodName: string) => createSelector(
  selectMethods,
  (methods): XAIMethodConfig | null => methods[methodName] || null
);

export const selectMethodConfigWithDefault = (methodName: string, defaultConfig: XAIMethodConfig) => createSelector(
  selectMethods,
  (methods): XAIMethodConfig => methods[methodName] || defaultConfig
);

export const selectConfigWithLoading = createSelector(
  selectConfig,
  selectConfigLoading,
  (config, loading) => ({ config, loading })
);

export const selectIsConfigLoaded = createSelector(
  selectConfig,
  (config) => config !== null
);
