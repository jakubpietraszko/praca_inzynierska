import { inject, Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { WrapperUnsupervisedAnyActivations } from './wrapper-unsupervised-any-activations/wrapper-unsupervised-any-activations';
import { constants } from '../../../constants';
import { AllLayersResponse } from '../../generated-interfaces/model_layers';

@Injectable({
  providedIn: 'root',
})
export class DialogUnsupervisedAnyActivations {
  dialog = inject(MatDialog);

  open(model: string, ref: string, layers: AllLayersResponse | null) {
    this.dialog.open(WrapperUnsupervisedAnyActivations, {
      width: constants.DIALOG_WIDTH,
      height: constants.DIALOG_HEIGHT,
      panelClass: constants.DIALOG_PANEL_CLASS,
      data: { model, ref, layers },
    });
  }
}
