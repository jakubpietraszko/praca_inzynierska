import { createAction, props } from '@ngrx/store';
import { ConfigModel, XAIMethodConfig } from './config.model';

export const loadConfig = createAction('[Config] Load Config');

export const loadConfigSuccess = createAction(
  '[Config] Load Config Success',
  props<{ config: ConfigModel }>()
);

export const loadConfigFailure = createAction(
  '[Config] Load Config Failure',
  props<{ error: string }>()
);

export const saveConfig = createAction(
  '[Config] Save Config',
  props<{ config: ConfigModel }>()
);

export const saveConfigSuccess = createAction(
  '[Config] Save Config Success',
  props<{ config: ConfigModel }>()
);

export const saveConfigFailure = createAction(
  '[Config] Save Config Failure',
  props<{ error: string }>()
);

export const updateConfig = createAction(
  '[Config] Update Config',
  props<{ config: Partial<ConfigModel> }>()
);

export const updateMethodConfig = createAction(
  '[Config] Update Method Config',
  props<{ methodName: string; methodConfig: Partial<XAIMethodConfig> }>()
);

export const resetConfig = createAction('[Config] Reset Config');
