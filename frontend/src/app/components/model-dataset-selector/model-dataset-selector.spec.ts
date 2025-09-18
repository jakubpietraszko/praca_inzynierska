import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelDatasetSelector } from './model-dataset-selector';

describe('ModelDatasetSelector', () => {
  let component: ModelDatasetSelector;
  let fixture: ComponentFixture<ModelDatasetSelector>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelDatasetSelector]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelDatasetSelector);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
