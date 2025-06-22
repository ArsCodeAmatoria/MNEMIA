# Conscious Core Dockerfile for MNEMIA (Haskell)
FROM fpco/stack-build:lts-21.25 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    netbase \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy stack configuration files
COPY stack.yaml stack.yaml.lock package.yaml ./

# Install GHC and dependencies (cached layer)
RUN stack setup
RUN stack build --dependencies-only

# Copy source code
COPY . .

# Build the application
RUN stack build --copy-bins --local-bin-path /app/bin

# Runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    netbase \
    libgmp10 \
    libnuma1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/bin/conscious-core-exe ./conscious-core

# Change ownership to app user
RUN chown -R app:app /app
USER app

EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["./conscious-core"] 