import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelSelector } from './model-selector';

describe('ModelSelector', () => {
  let component: ModelSelector;
  let fixture: ComponentFixture<ModelSelector>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelSelector]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelSelector);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
