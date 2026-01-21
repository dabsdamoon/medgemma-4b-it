import React, { useState } from 'react';
import { Container, Row, Col, Form, Button, Card, Spinner, Alert } from 'react-bootstrap';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
    } else {
      setImage(null);
      setImagePreview(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResponse('');

    const formData = new FormData();
    formData.append('text', text);
    if (image) {
      formData.append('image', image);
    }

    try {
      // Since we set up the proxy in vite.config.js, we can just hit /process
      // But if running separately, use the full URL. 
      // Using full URL for robustness if proxy isn't used.
      const apiEndpoint = 'http://127.0.0.1:8000/process'; 
      
      const res = await axios.post(apiEndpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (res.data.error) {
        setError(res.data.error);
      } else {
        setResponse(res.data.response);
      }
    } catch (err) {
      console.error(err);
      setError('An error occurred while communicating with the backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-vh-100 bg-light py-5">
      <Container>
        <Row className="justify-content-center">
          <Col md={10} lg={8}>
            <Card className="shadow-sm border-0">
              <Card.Header className="bg-primary text-white p-4">
                <h2 className="mb-0 text-center">MedGemma Medical Assistant</h2>
                <p className="text-center mb-0 opacity-75">AI-Powered Medical Document Summarization & Analysis</p>
              </Card.Header>
              <Card.Body className="p-4">
                <Form onSubmit={handleSubmit}>
                  <Form.Group className="mb-4">
                    <Form.Label className="fw-bold">Medical Text / Query</Form.Label>
                    <Form.Control
                      as="textarea"
                      rows={6}
                      placeholder="Enter patient notes, symptoms, or medical text here..."
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      required={!image} // Require text if no image, or make text mandatory? Let's make text mandatory for the prompt context.
                    />
                  </Form.Group>

                  <Form.Group className="mb-4">
                    <Form.Label className="fw-bold">Medical Image (Optional)</Form.Label>
                    <Form.Control
                      type="file"
                      accept="image/*"
                      onChange={handleImageChange}
                    />
                    {imagePreview && (
                      <div className="mt-3 text-center">
                        <img
                          src={imagePreview}
                          alt="Preview"
                          className="img-fluid rounded border"
                          style={{ maxHeight: '300px' }}
                        />
                      </div>
                    )}
                  </Form.Group>

                  {error && <Alert variant="danger">{error}</Alert>}

                  <div className="d-grid gap-2">
                    <Button variant="primary" size="lg" type="submit" disabled={loading}>
                      {loading ? (
                        <>
                          <Spinner
                            as="span"
                            animation="border"
                            size="sm"
                            role="status"
                            aria-hidden="true"
                            className="me-2"
                          />
                          Analyzing...
                        </>
                      ) : (
                        'Analyze Document'
                      )}
                    </Button>
                  </div>
                </Form>
              </Card.Body>
            </Card>

            {response && (
              <Card className="mt-4 shadow-sm border-0">
                <Card.Header className="bg-success text-white">
                  <h4 className="mb-0">Analysis Result</h4>
                </Card.Header>
                <Card.Body className="p-4">
                  <div style={{ whiteSpace: 'pre-wrap' }}>
                    {response}
                  </div>
                </Card.Body>
              </Card>
            )}
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;