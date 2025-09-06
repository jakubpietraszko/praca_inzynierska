import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CompareHeatmaps } from './compare-heatmaps';

describe('CompareHeatmaps', () => {
  let component: CompareHeatmaps;
  let fixture: ComponentFixture<CompareHeatmaps>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CompareHeatmaps]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CompareHeatmaps);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
