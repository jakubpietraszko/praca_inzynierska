import { ActionReducerMap, MetaReducer } from '@ngrx/store';
import { environment } from '../../environments/environment';
import { AppState } from './app.state';
import { configReducer } from './config/config.reducer';

export const reducers: ActionReducerMap<AppState> = {
  config: configReducer,
};

export const metaReducers: MetaReducer<AppState>[] = !environment.production
  ? []
  : [];
