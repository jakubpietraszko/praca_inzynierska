import { inject, Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { WrapperUnsupervisedDataset } from './wrapper-unsupervised-dataset/wrapper-unsupervised-dataset';
import { constants } from '../../../../constants';

@Injectable({
  providedIn: 'root',
})
export class DialogUnsupervisedDataset {
  dialog = inject(MatDialog);

  open(dataset: string) {
    this.dialog.open(WrapperUnsupervisedDataset, {
      width: constants.DIALOG_WIDTH,
      height: constants.DIALOG_HEIGHT,
      panelClass: constants.DIALOG_PANEL_CLASS,
      data: {
        dataset,
      },
    });
  }
}
