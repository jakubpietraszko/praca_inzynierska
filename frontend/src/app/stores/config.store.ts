import { computed, inject } from '@angular/core';
import { patchState, signalStore, withComputed, withMethods, withState } from '@ngrx/signals';
import { rxMethod } from '@ngrx/signals/rxjs-interop';
import { pipe, switchMap, tap, catchError, of } from 'rxjs';
import { ConfigService } from '../services/config.service';
import { ConfigModel, XAIMethodConfig } from '../state/config/config.model';

interface ConfigState {
  config: ConfigModel | null;
  loading: boolean;
  error: string | null;
}

const initialState: ConfigState = {
  config: null,
  loading: false,
  error: null,
};

export const ConfigStore = signalStore(
  { providedIn: 'root' },
  withState(initialState),
  withComputed(({ config }) => ({
    methods: computed(() => config()?.methods || {}),
    methodNames: computed(() => Object.keys(config()?.methods || {})),
    isConfigLoaded: computed(() => config() !== null),
    configWithLoading: computed(() => ({
      config: config(),
      loading: config() !== null,
    })),
  })),
  withMethods((store, configService = inject(ConfigService)) => ({
    loadConfig: rxMethod<void>(
      pipe(
        tap(() => patchState(store, { loading: true, error: null })),
        switchMap(() =>
          configService.getConfig().pipe(
            tap((config) =>
              patchState(store, { config, loading: false, error: null })
            ),
            catchError((error) => {
              console.error('Failed to load config:', error);
              patchState(store, {
                loading: false,
                error: error.message || 'Failed to load config',
              });
              return of(null);
            })
          )
        )
      )
    ),

    saveConfig: rxMethod<ConfigModel>(
      pipe(
        tap(() => patchState(store, { loading: true, error: null })),
        switchMap((config) =>
          configService.saveConfig(config).pipe(
            tap(() =>
              patchState(store, { config, loading: false, error: null })
            ),
            catchError((error) => {
              console.error('Failed to save config:', error);
              patchState(store, {
                loading: false,
                error: error.message || 'Failed to save config',
              });
              return of(null);
            })
          )
        )
      )
    ),

    updateConfig: (configUpdate: Partial<ConfigModel>) => {
      const currentConfig = store.config();
      if (currentConfig) {
        patchState(store, {
          config: { ...currentConfig, ...configUpdate },
        });
      }
    },

    updateMethodConfig: (methodName: string, methodConfig: Partial<XAIMethodConfig>) => {
      const currentConfig = store.config();
      if (currentConfig) {
        patchState(store, {
          config: {
            ...currentConfig,
            methods: {
              ...currentConfig.methods,
              [methodName]: {
                ...(currentConfig.methods[methodName] || {}),
                ...methodConfig,
              },
            },
          },
        });
      }
    },

    resetConfig: () => patchState(store, initialState),

    getMethodConfig: (methodName: string): XAIMethodConfig | null => {
      const methods = store.methods();
      return methods[methodName] || null;
    },

    getMethodConfigWithDefault: (methodName: string, defaultConfig: XAIMethodConfig): XAIMethodConfig => {
      const methods = store.methods();
      return methods[methodName] || defaultConfig;
    },
  }))
);
