import { inject, Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { WrapperDatasetsBalance } from './wrapper-datasets-balance/wrapper-datasets-balance';
import { constants } from '../../../../constants';

@Injectable({
  providedIn: 'root',
})
export class DialogDatasetsBalance {
  dialog = inject(MatDialog);

  open(dataset: string) {
    this.dialog.open(WrapperDatasetsBalance, {
      width: constants.DIALOG_WIDTH,
      height: constants.DIALOG_HEIGHT,
      panelClass: constants.DIALOG_PANEL_CLASS,
      data: {
        dataset,
      },
    });
  }
}
