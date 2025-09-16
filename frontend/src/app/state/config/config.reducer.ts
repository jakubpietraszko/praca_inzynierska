import { createReducer, on } from '@ngrx/store';
import { ConfigState } from './config.model';
import * as ConfigActions from './config.actions';

export const initialState: ConfigState = {
  config: null,
  loading: false,
  error: null
};

export const configReducer = createReducer(
  initialState,
  
  on(ConfigActions.loadConfig, (state) => ({
    ...state,
    loading: true,
    error: null
  })),
  
  on(ConfigActions.loadConfigSuccess, (state, { config }) => ({
    ...state,
    config,
    loading: false,
    error: null
  })),
  
  on(ConfigActions.loadConfigFailure, (state, { error }) => ({
    ...state,
    loading: false,
    error
  })),
  
  on(ConfigActions.saveConfig, (state) => ({
    ...state,
    loading: true,
    error: null
  })),
  
  on(ConfigActions.saveConfigSuccess, (state, { config }) => ({
    ...state,
    config,
    loading: false,
    error: null
  })),
  
  on(ConfigActions.saveConfigFailure, (state, { error }) => ({
    ...state,
    loading: false,
    error
  })),
  
  on(ConfigActions.updateConfig, (state, { config: configUpdate }) => ({
    ...state,
    config: state.config ? { ...state.config, ...configUpdate } : null
  })),

  on(ConfigActions.updateMethodConfig, (state, { methodName, methodConfig }) => ({
    ...state,
    config: state.config ? {
      ...state.config,
      methods: {
        ...state.config.methods,
        [methodName]: {
          ...state.config.methods[methodName],
          ...methodConfig
        }
      }
    } : null
  })),
  
  on(ConfigActions.resetConfig, () => initialState)
);
