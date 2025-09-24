import { Injectable, inject } from '@angular/core';
import { Actions, createEffect, ofType } from '@ngrx/effects';
import { of } from 'rxjs';
import { map, catchError, switchMap } from 'rxjs/operators';
import * as ConfigActions from './config.actions';
import { ConfigService } from '../../services/config.service';
import { ConfigModel } from './config.model';

@Injectable()
export class ConfigEffects {
  private actions$ = inject(Actions);
  private configService = inject(ConfigService);

  loadConfig$ = createEffect(() =>
    this.actions$.pipe(
      ofType(ConfigActions.loadConfig),
      switchMap(() =>
        this.configService.getConfig().pipe(
          map((config: ConfigModel) =>
            ConfigActions.loadConfigSuccess({ config })
          ),
          catchError((error) => {
            console.error('Failed to load config:', error);
            return of(
              ConfigActions.loadConfigFailure({
                error: error.message || 'Failed to load config',
              })
            );
          })
        )
      )
    )
  );

  saveConfig$ = createEffect(() =>
    this.actions$.pipe(
      ofType(ConfigActions.saveConfig),
      switchMap(({ config }) =>
        this.configService.saveConfig(config).pipe(
          map((response: any) =>
            ConfigActions.saveConfigSuccess({ config })
          ),
          catchError((error) =>
            of(
              ConfigActions.saveConfigFailure({
                error: error.message || 'Failed to save config',
              })
            )
          )
        )
      )
    )
  );
}
