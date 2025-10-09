import {
  patchState,
  signalStore,
  withComputed,
  withMethods,
  withState,
} from '@ngrx/signals';
import { ConstantState } from './constants.interface';
import { computed, inject } from '@angular/core';
import { ConstantsService } from './constants.service';
import { rxMethod } from '@ngrx/signals/rxjs-interop';
import { pipe, switchMap, tap, catchError, of } from 'rxjs';

const initialState: ConstantState = {
  constants: null,
  loading: false,
  error: null,
};

export const ConstantStore = signalStore(
  { providedIn: 'root' },
  withState(initialState),
  withComputed(({ constants }) => ({
    translations: computed(() => constants()?.translations ?? {}),
    isConstantsLoaded: computed(() => constants() !== null),
  })),
  withMethods((store, constantsService = inject(ConstantsService)) => ({
    loadConstants: rxMethod<void>(
      pipe(
        tap(() => patchState(store, { loading: true, error: null })),
        switchMap(() =>
          constantsService.getConstants().pipe(
            tap((constants) => {
              patchState(store, { constants, loading: false, error: null });
            }),
            catchError((error) => {
              patchState(store, {
                loading: false,
                error: error.message || 'Failed to load constants',
              });
              return of(null);
            }),
          ),
        ),
      ),
    ),
  })),
);
