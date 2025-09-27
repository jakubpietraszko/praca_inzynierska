import { inject, Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { constants } from '../../../../constants';
import { WrapperUnsupervisedActivations } from './wrapper-unsupervised-activations/wrapper-unsupervised-activations';

@Injectable({
  providedIn: 'root',
})
export class DialogUnsupervisedActivations {
  dialog = inject(MatDialog);

  open() {
    this.dialog.open(WrapperUnsupervisedActivations, {
      width: constants.DIALOG_WIDTH,
      height: constants.DIALOG_HEIGHT,
      panelClass: constants.DIALOG_PANEL_CLASS,
    });
  }
}
