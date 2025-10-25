import {
  Component,
  inject,
  OnDestroy,
  OnInit,
  signal,
  computed,
} from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { DatasetsBalance } from '../datasets-balance';
import { Subscription } from 'rxjs';
import { DatasetBalanceResponse } from '../../../../generated-interfaces/datasets_balance';
import { BarPlot } from '../../../shared/bar-plot/bar-plot';

@Component({
  selector: 'app-wrapper-datasets-balance',
  imports: [BarPlot],
  templateUrl: './wrapper-datasets-balance.html',
  styleUrl: './wrapper-datasets-balance.css',
})
export class WrapperDatasetsBalance implements OnInit, OnDestroy {
  data = inject(MAT_DIALOG_DATA) as {
    dataset: string;
  };

  datasetsBalance = inject(DatasetsBalance);

  $balance = signal<Record<string, number> | null>(null);

  $balanceValues = computed(() => {
    const balance = this.$balance();
    return balance ? Object.values(balance) : [];
  });

  $balanceLabels = computed(() => {
    const balance = this.$balance();
    return balance ? Object.keys(balance) : [];
  });

  $showPlot = computed(() => {
    const balance = this.$balance();
    return balance !== null && Object.keys(balance).length > 0;
  });

  subscription = new Subscription();

  ngOnInit(): void {
    this.subscription.add(
      this.datasetsBalance
        .getDatasetBalance(this.data.dataset)
        .subscribe((response: DatasetBalanceResponse) => {
          this.$balance.set(response.balance);
        }),
    );
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }
}
