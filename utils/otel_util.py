# utils/otel_util.py
import inspect
import uuid
import logging
from functools import wraps
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from utils.traceable import Traceable  # Import Traceable Interface

# Initialize OpenTelemetry Tracer Provider
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)

# Initialize OpenTelemetry Meter Provider for Prometheus
reader = PrometheusMetricReader()
meter_provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(meter_provider)

# Get Tracer and Meter
def get_tracer(name: str):
    return trace.get_tracer(name)

def get_meter(name: str):
    return metrics.get_meter(name)

# Enable OpenTelemetry Logging
LoggingInstrumentor().instrument()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define OpenTelemetry Metrics
GENERIC_COUNTER = get_meter(__name__).create_counter(
    "generic_method_calls",
    "Tracks method calls with parameters",
    unit="1"
)

def extract_traceable_data(obj):
    """Extract trace data from objects that implement Traceable."""
    if isinstance(obj, Traceable):
        return obj.get_trace_data()
    elif isinstance(obj, list):  # Handle lists of Traceable objects
        return [extract_traceable_data(item) for item in obj if isinstance(item, Traceable)]
    return None

def trace_and_log_metrics(func):
    """Decorator to trace execution and automatically log method parameters as OpenTelemetry traces."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        method_name = func.__name__
        tracer = get_tracer(self.__class__.__name__)

        with tracer.start_as_current_span(method_name) as span:
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            param_data = {k: v for k, v in bound_args.arguments.items() if k != 'self'}
            experiment_id = getattr(self, "experiment_id", str(uuid.uuid4()))

            span.set_attribute("experiment_id", experiment_id)
            span.set_attribute("method_name", method_name)
            
            # Extract traceable metadata from inputs
            for key, value in param_data.items():
                span.set_attribute(key, value)
                traceable_metadata = extract_traceable_data(value)
                if traceable_metadata:
                    span.set_attribute(f"{key}_traceable", str(traceable_metadata))  # Store as string

            try:
                result = func(self, *args, **kwargs)

                # Extract traceable metadata from output
                traceable_metadata = extract_traceable_data(result)
                if traceable_metadata:
                    span.set_attribute("return_traceable", str(traceable_metadata))

                GENERIC_COUNTER.add(1, {"method": method_name})
                logger.info(f"Experiment {experiment_id} - Successfully executed {method_name}")
                return result

            except Exception as e:
                logger.error(f"Experiment {experiment_id} - Error in {method_name}: {e}")
                raise

    return wrapper