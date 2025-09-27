import { inject, Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { constants } from '../../../../constants';
import { WrapperUnsupervisedHeatmaps } from './wrapper-unsupervised-heatmaps/wrapper-unsupervised-heatmaps';

@Injectable({
  providedIn: 'root',
})
export class DialogUnsupervisedHeatmaps {
  dialog = inject(MatDialog);

  open(dataset: string, label: string) {
    this.dialog.open(WrapperUnsupervisedHeatmaps, {
      width: constants.DIALOG_WIDTH,
      height: constants.DIALOG_HEIGHT,
      panelClass: constants.DIALOG_PANEL_CLASS,
      data: { dataset, label },
    });
  }
}
