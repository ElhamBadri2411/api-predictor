# Assumptions and Design Decisions

## Key Assumptions

- Used synthetic training data since no real API logs were available
- Recent 3-5 events are most predictive of next user actions
- Users follow logical API workflows (list → detail → update)
- OpenAPI json specs are accessible and well-formed

## Trade-offs Made

- Logistic Regression over Random Forest: RF showed better accuracy on training data but was slower
- Conservative safety: Block DELETE operations unless explicitly requested
- Synthetic data: Rapid development vs perfect realism

## Future Work

- Train on real user interaction data
- Add user feedback collection
- Support multi-step predictions
