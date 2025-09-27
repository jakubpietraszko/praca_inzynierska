import { inject, Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { constants } from '../../../../constants';
import { WrapperCompareHeatmaps } from './wrapper-compare-heatmaps/wrapper-compare-heatmaps';

@Injectable({
  providedIn: 'root',
})
export class DialogCompareHeatmaps {
  dialog = inject(MatDialog);

  open() {
    this.dialog.open(WrapperCompareHeatmaps, {
      width: constants.DIALOG_WIDTH,
      height: constants.DIALOG_HEIGHT,
      panelClass: constants.DIALOG_PANEL_CLASS,
    });
  }
}
