import { inject, Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { WrapperUnsupervisedLabel } from './wrapper-unsupervised-label/wrapper-unsupervised-label';
import { constants } from '../../../../constants';

@Injectable({
  providedIn: 'root',
})
export class DialogUnsupervisedLabel {
  dialog = inject(MatDialog);

  open(dataset: string, label: string) {
    this.dialog.open(WrapperUnsupervisedLabel, {
      width: constants.DIALOG_WIDTH,
      height: constants.DIALOG_HEIGHT,
      panelClass: constants.DIALOG_PANEL_CLASS,
      data: {
        dataset,
        label,
      },
    });
  }
}
