from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

import sys
import os

sys.path.append(os.getcwd())

from app.core.config import settings
from app.db import Base

from app.models.user import User
from app.models.concept import ConceptTaxonomy
from app.models.course import Course
from app.models.document_chunk import DocumentChunk, ChunkDomainTag

from app.models.employee_profile import EmployeeProfile
from app.models.employee_training_history import EmployeeTrainingHistory
from app.models.competency_matrix import CompetencyFrameworkMatrix
from app.models.mcq import MCQ
from app.models.assessment_history import AssessmentHistory
from app.models.concept_mastery import ConceptMastery

from app.models.recommendation import Recommendation
from app.models.concept_review_queue import ConceptReviewQueue
from app.models.mcq_generation_log import McqGenerationLog


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# Add all SQLAlchemy models to Base.metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""

    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""

    connectable = engine_from_config(
        config.get_section(
            config.config_ini_section,
            {}
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:

        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()